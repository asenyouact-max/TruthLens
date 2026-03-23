"""
ClaudeReasoner — sends all signal results to Claude API for
multi-signal reasoning, explanation generation, and final verdict.

Claude's job here is NOT to replace the models — it reasons across
their outputs, catches cross-signal contradictions, contextualizes
anomalies, and generates a human-readable explanation.
"""
import json
import anthropic
from dataclasses import dataclass
from typing import List, Optional, Any
from config import settings


@dataclass
class ClaudeOutput:
    explanation: str
    key_observations: List[str]
    confidence_rationale: str


class ClaudeReasoner:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def reason(
        self,
        session_id: str,
        is_ai_generated: bool,
        deepfake_breakdown: Any,
        signal_breakdown: Optional[Any],
        facial_detail: Optional[Any],
        pose_detail: Optional[Any],
        voice_detail: Optional[Any],
        nlp_detail: Optional[Any],
        duration: float,
    ) -> ClaudeOutput:

        if is_ai_generated:
            prompt = self._build_deepfake_prompt(deepfake_breakdown, duration)
        else:
            prompt = self._build_deception_prompt(
                deepfake_breakdown, signal_breakdown,
                facial_detail, pose_detail, voice_detail, nlp_detail, duration
            )

        message = await self.client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=1000,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text
        return self._parse_response(raw)

    def _system_prompt(self) -> str:
        return """You are a forensic AI analyst for TruthLens, an app that detects 
deception and AI-generated video. You receive structured signal data from multiple 
detection models and your job is to:

1. Reason across all signals holistically — not just report numbers
2. Identify which signals are most meaningful vs noise
3. Flag cross-signal confirmations (e.g. facial stress AND voice pitch elevation = stronger)
4. Note any signals that contradict each other and explain why
5. Generate a clear, non-accusatory explanation for the user
6. Always remind the user this is probabilistic, not proof

Respond in this exact JSON format:
{
  "explanation": "2-4 sentence plain English summary for the user",
  "key_observations": ["observation 1", "observation 2", "observation 3"],
  "confidence_rationale": "1 sentence on why confidence is high/medium/low"
}

Be measured and fair. Avoid sensational language. Never say the person is lying — 
say signals are elevated, patterns are consistent with stress, etc."""

    def _build_deepfake_prompt(self, deepfake, duration: float) -> str:
        has_face = getattr(deepfake, "_has_face", True)
        flags = getattr(deepfake, "_flags", [])
        return f"""ANALYSIS TYPE: AI-Generated Video Detection (full-frame analysis)

Duration: {duration:.1f}s
Face detected: {has_face}

ZONE-BY-ZONE FORENSIC SCORES (0=natural, 1=AI-generated):
- Face region (rPPG, texture, blink):    {deepfake.biological_signals_score:.3f}
- Hands (finger anatomy, morphing):      (embedded in visual artifacts)
- Background (smoothness, edge seams):   {deepfake.visual_artifacts_score:.3f}
- Objects + text (garbled text, physics):{deepfake.frequency_artifacts_score:.3f}
- Body + clothing (proportions, drift):  {deepfake.temporal_consistency_score:.3f}
- Whole frame (FFT, noise, color):       {deepfake.frequency_artifacts_score:.3f}
- Audio (TTS/clone, room acoustics):     {deepfake.av_sync_score:.3f}
- Metadata (device fingerprint, EXIF):   {deepfake.metadata_score:.3f}
- ENSEMBLE SCORE:                        {deepfake.ensemble_score:.3f}

FLAGS RAISED: {flags if flags else 'none'}

NOTE: This analysis covers the full video frame — not just the face.
Even videos with no human present are analyzed for AI generation artifacts.

Please explain which zones were most decisive, note any surprising signal
combinations, and provide your explanation + key observations."""

    def _build_deception_prompt(
        self, deepfake, signals, facial, pose, voice, nlp, duration: float
    ) -> str:
        face_flags = facial.flags if facial else []
        pose_flags = pose.flags if pose else []
        voice_flags = voice.flags if voice else []
        nlp_flags = nlp.flags if nlp else []
        transcript = (voice.transcript[:500] if voice and voice.transcript else "Not available")

        return f"""ANALYSIS TYPE: Deception Detection (Video confirmed as real)

Duration: {duration:.1f}s

AI DETECTION (passed gate):
- Ensemble deepfake score: {deepfake.ensemble_score:.3f} (below threshold — confirmed real)

FACIAL ANALYSIS:
- Stress score: {signals.facial_au_score:.3f if signals else 'N/A'}
- Gaze aversion rate: {facial.gaze_aversion_rate:.3f if facial else 'N/A'}
- Smile asymmetry: {facial.smile_asymmetry:.3f if facial else 'N/A'}
- Micro-expressions: {facial.micro_expression_count if facial else 'N/A'}
- Flags: {face_flags}

BODY LANGUAGE:
- Deception score: {signals.body_language_score:.3f if signals else 'N/A'}
- Self-touch rate: {pose.self_touch_rate:.3f if pose else 'N/A'}
- Posture shifts: {pose.posture_shift_count if pose else 'N/A'}
- Flags: {pose_flags}

VOICE PROSODY:
- Anomaly score: {signals.voice_prosody_score:.3f if signals else 'N/A'}
- Pitch mean: {voice.pitch_mean:.1f if voice else 'N/A'} Hz
- Pause rate: {voice.pause_rate:.1f if voice else 'N/A'} /min
- Filler words: {voice.filler_word_rate:.1f if voice else 'N/A'} /min
- Flags: {voice_flags}

LINGUISTIC ANALYSIS:
- Deception score: {signals.nlp_score:.3f if signals else 'N/A'}
- Distancing language: {nlp.distancing_score:.3f if nlp else 'N/A'}
- Negation rate: {nlp.negation_rate:.3f if nlp else 'N/A'}
- Contradictions: {nlp.contradiction_count if nlp else 'N/A'}
- Flags: {nlp_flags}

TRANSCRIPT EXCERPT:
{transcript}

Please reason across all signals holistically. Note convergences and 
contradictions between channels. Generate explanation + key observations."""

    def _parse_response(self, raw: str) -> ClaudeOutput:
        try:
            # Strip any markdown code fences
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean)
            return ClaudeOutput(
                explanation=data.get("explanation", raw),
                key_observations=data.get("key_observations", []),
                confidence_rationale=data.get("confidence_rationale", ""),
            )
        except Exception:
            # Fallback: return raw as explanation
            return ClaudeOutput(
                explanation=raw[:500],
                key_observations=[],
                confidence_rationale="Structured parsing failed — raw response returned",
            )
