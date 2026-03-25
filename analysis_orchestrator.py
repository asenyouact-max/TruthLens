"""
AnalysisOrchestrator — runs all detection pipelines in parallel,
then sends structured results to Claude for final reasoning.
"""
import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from full_frame_detector import FullFrameAIDetector as DeepfakeDetector
from facial_analyzer import FacialAnalyzer
from pose_analyzer import PoseAnalyzer
from voice_analyzer import VoiceAnalyzer
from nlp_analyzer import NLPAnalyzer
from claude_reasoner import ClaudeReasoner
from multi_brain_reasoner import MultiBrainReasoner
from multi_brain_reasoner import MultiBrainReasoner
from config import settings
from video_processor import VideoProcessor
from schemas import (
    AnalysisResult, DeepfakeBreakdown, SignalBreakdown
)
from config import settings
from redis_client import set_session_state


class AnalysisOrchestrator:
    def __init__(
        self,
        session_id: str,
        db: AsyncSession,
        video_path: Optional[str] = None,
    ):
        self.session_id = session_id
        self.db = db
        self.video_path = video_path

        self.deepfake = DeepfakeDetector()
        self.facial = FacialAnalyzer()
        self.pose = PoseAnalyzer()
        self.voice = VoiceAnalyzer()
        self.nlp = NLPAnalyzer()
        self.claude = MultiBrainReasoner() if settings.MULTI_BRAIN_ENABLED else ClaudeReasoner()
        self.processor = VideoProcessor()

    async def run_full_pipeline(self) -> AnalysisResult:
        # 1. Preprocess video into frames + audio
        frames, audio_path, duration, fps = await self.processor.extract(self.video_path)

        # 2. Run DEEPFAKE DETECTION first (gate check) — all detectors in parallel
        deepfake_result = await self.deepfake.analyze(frames, audio_path, self.video_path)

        is_ai = deepfake_result.ensemble_score >= settings.DEEPFAKE_CONFIDENCE_THRESHOLD

        signal_breakdown = None
        deception_score = None
        deception_confidence = None

        # 3. Only run lie detection if video passes as real
        if not is_ai:
            facial_task = asyncio.create_task(self.facial.analyze(frames))
            pose_task = asyncio.create_task(self.pose.analyze(frames))
            voice_task = asyncio.create_task(self.voice.analyze(audio_path))
            nlp_task = asyncio.create_task(self.nlp.analyze(audio_path))

            facial_result, pose_result, voice_result, nlp_result = await asyncio.gather(
                facial_task, pose_task, voice_task, nlp_task
            )

            signal_breakdown = SignalBreakdown(
                facial_au_score=facial_result.stress_score,
                body_language_score=pose_result.deception_score,
                voice_prosody_score=voice_result.anomaly_score,
                nlp_score=nlp_result.deception_score,
            )

            # Weighted average deception score
            deception_score = (
                signal_breakdown.facial_au_score * 0.30
                + signal_breakdown.body_language_score * 0.25
                + signal_breakdown.voice_prosody_score * 0.25
                + signal_breakdown.nlp_score * 0.20
            )
            deception_confidence = min(
                facial_result.confidence,
                pose_result.confidence,
                voice_result.confidence,
                nlp_result.confidence,
            )
        else:
            facial_result = pose_result = voice_result = nlp_result = None

        # 4. Send everything to Claude for final reasoning
        claude_output = await self.claude.reason(
            session_id=self.session_id,
            is_ai_generated=is_ai,
            deepfake_breakdown=deepfake_result,
            signal_breakdown=signal_breakdown,
            facial_detail=facial_result,
            pose_detail=pose_result,
            voice_detail=voice_result,
            nlp_detail=nlp_result,
            duration=duration,
        )

        result = AnalysisResult(
            session_id=self.session_id,
            is_ai_generated=is_ai,
            ai_confidence=deepfake_result.ensemble_score,
            deepfake_breakdown=deepfake_result,
            deception_score=deception_score,
            deception_confidence=deception_confidence,
            signal_breakdown=signal_breakdown,
            explanation=claude_output.explanation,
            key_observations=claude_output.key_observations,
            duration_seconds=duration,
            frames_analyzed=len(frames),
            created_at=datetime.utcnow(),
        )

        # Cache result in Redis
        await set_session_state(self.session_id, {"result": result.model_dump(mode="json")})

        return result
