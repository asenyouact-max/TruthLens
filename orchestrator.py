"""
Analysis orchestrator.
Coordinates all detectors in parallel then calls Claude for fusion.
"""
import asyncio
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import structlog

from facial import FacialDetector
from voice import VoiceDetector
from nlp import NLPDetector
from deepfake import DeepfakeDetector
from claude_fusion import run_fusion
from config import settings

log = structlog.get_logger()

_executor = ThreadPoolExecutor(max_workers=4)

# Singleton detectors (loaded once)
_facial = FacialDetector()
_voice = VoiceDetector(whisper_model=settings.WHISPER_MODEL)
_nlp = NLPDetector()
_deepfake = DeepfakeDetector(threshold=settings.DEEPFAKE_THRESHOLD)


async def _run_in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def analyze_video(
    video_path: str,
    audio_path: Optional[str] = None,
    metadata: Optional[dict] = None,
    baseline: Optional[dict] = None,
) -> dict:
    log.info("Starting full analysis pipeline", video=video_path)

    # Extract audio if not provided separately
    if not audio_path:
        audio_path = await _run_in_thread(_extract_audio, video_path)

    # Run all detectors in parallel
    facial_task = _run_in_thread(_facial.process_video, video_path)
    voice_task = _run_in_thread(_voice.analyze_audio, audio_path) if audio_path else asyncio.sleep(0)
    deepfake_task = _run_in_thread(_deepfake.analyze_video, video_path, metadata or {})

    facial_result, voice_result, deepfake_result = await asyncio.gather(
        facial_task, voice_task, deepfake_task
    )

    # NLP needs transcript from voice result — run after
    nlp_result = None
    if voice_result and hasattr(voice_result, "transcript"):
        nlp_result = await _run_in_thread(_nlp.analyze, voice_result.transcript)

    # Claude fusion — async
    fusion = await run_fusion(
        facial=facial_result,
        voice=voice_result,
        nlp=nlp_result,
        deepfake=deepfake_result,
        baseline=baseline,
    )

    def _to_dict(obj):
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(obj)
        return obj

    return {
        "deception_score": fusion.get("deception_score", 0.0),
        "is_ai_generated": deepfake_result.is_ai_generated if deepfake_result else False,
        "ai_confidence": deepfake_result.confidence if deepfake_result else 0.0,
        "verdict": fusion.get("verdict", "inconclusive"),
        "confidence": fusion.get("confidence", 0.0),
        "reasoning": fusion.get("reasoning", ""),
        "key_indicators": fusion.get("key_indicators", []),
        "confounding_factors": fusion.get("confounding_factors", []),
        "signal_weights": fusion.get("signal_weights", {}),
        "signals": {
            "facial": _to_dict(facial_result),
            "voice": _to_dict(voice_result),
            "nlp": _to_dict(nlp_result),
            "deepfake": _to_dict(deepfake_result),
        },
    }


def _extract_audio(video_path: str) -> Optional[str]:
    try:
        import ffmpeg
        audio_path = video_path.replace(".mp4", ".wav").replace(".mov", ".wav")
        if not audio_path.endswith(".wav"):
            audio_path += ".wav"
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec="pcm_s16le", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=True)
        )
        return audio_path
    except Exception as e:
        log.error("Audio extraction failed", error=str(e))
        return None
