"""
VoiceAnalyzer — extracts deception-related prosodic features:
- Pitch (F0) elevation and variance (stress = higher pitch)
- Speech rate changes (deceptive = slower or faster than baseline)
- Pause patterns (longer pre-response pauses)
- Voice tremor / jitter
- Filler word rate (um, uh, like)
- Energy envelope variance

Uses librosa for feature extraction + Whisper for transcription.
"""
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VoiceResult:
    anomaly_score: float        # 0-1 overall prosodic anomaly
    confidence: float
    pitch_mean: float
    pitch_variance: float
    speech_rate_wpm: float
    pause_rate: float           # pauses per minute
    filler_word_rate: float     # fillers per minute
    tremor_score: float
    transcript: Optional[str]
    flags: List[str]


class VoiceAnalyzer:
    FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "actually"}

    async def analyze(self, audio_path: str) -> VoiceResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, audio_path)

    def _analyze_sync(self, audio_path: str) -> VoiceResult:
        import os
        if not os.path.exists(audio_path):
            return self._fallback_result("Audio file not found")

        try:
            import librosa
        except ImportError:
            return self._fallback_result("librosa not installed")

        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr

        # Pitch (F0) extraction
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_variance = float(np.var(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # Voice tremor: jitter in pitch
        tremor_score = self._calculate_tremor(voiced_f0)

        # Pause detection using RMS energy
        pause_rate, speech_ratio = self._detect_pauses(y, sr, duration)

        # Transcription via Whisper
        transcript, filler_rate, wpm = self._transcribe(audio_path, duration)

        # Anomaly scoring
        # High pitch + high variance + many pauses + fillers = higher deception signal
        pitch_score = min(1.0, max(0.0, (pitch_mean - 150) / 200.0)) if pitch_mean > 0 else 0.3
        variance_score = min(1.0, pitch_variance / 5000.0)
        pause_score = min(1.0, pause_rate / 20.0)
        filler_score = min(1.0, filler_rate / 10.0)

        anomaly_score = (
            pitch_score * 0.25
            + variance_score * 0.20
            + pause_score * 0.25
            + filler_score * 0.15
            + tremor_score * 0.15
        )

        flags = []
        if pitch_mean > 250:
            flags.append(f"Elevated pitch detected ({pitch_mean:.0f} Hz)")
        if pause_rate > 15:
            flags.append(f"High pause rate: {pause_rate:.1f} pauses/min")
        if filler_rate > 8:
            flags.append(f"High filler word rate: {filler_rate:.1f}/min")
        if tremor_score > 0.6:
            flags.append("Voice tremor detected")

        return VoiceResult(
            anomaly_score=round(anomaly_score, 3),
            confidence=0.80,
            pitch_mean=round(pitch_mean, 1),
            pitch_variance=round(pitch_variance, 1),
            speech_rate_wpm=round(wpm, 1),
            pause_rate=round(pause_rate, 2),
            filler_word_rate=round(filler_rate, 2),
            tremor_score=round(tremor_score, 3),
            transcript=transcript,
            flags=flags,
        )

    def _calculate_tremor(self, f0: np.ndarray) -> float:
        if len(f0) < 10:
            return 0.0
        diffs = np.abs(np.diff(f0))
        jitter = float(np.mean(diffs) / (np.mean(f0) + 1e-8))
        return min(1.0, jitter * 20)

    def _detect_pauses(self, y: np.ndarray, sr: int, duration: float):
        import librosa
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        silence_threshold = np.percentile(rms, 20)
        is_silent = rms < silence_threshold
        # Count transitions from speech to silence
        pauses = int(np.sum(np.diff(is_silent.astype(int)) == 1))
        pause_rate = pauses / (duration / 60.0) if duration > 0 else 0.0
        speech_ratio = float(np.mean(~is_silent))
        return pause_rate, speech_ratio

    def _transcribe(self, audio_path: str, duration: float):
        """Use Whisper for transcription + filler detection."""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            transcript = result["text"]
            words = transcript.lower().split()
            filler_count = sum(1 for w in words if w.strip(".,!?") in self.FILLER_WORDS)
            wpm = len(words) / (duration / 60.0) if duration > 0 else 0
            filler_rate = filler_count / (duration / 60.0) if duration > 0 else 0
            return transcript, filler_rate, wpm
        except Exception:
            return None, 0.0, 0.0

    def _fallback_result(self, reason: str) -> VoiceResult:
        return VoiceResult(
            anomaly_score=0.0, confidence=0.0, pitch_mean=0.0,
            pitch_variance=0.0, speech_rate_wpm=0.0, pause_rate=0.0,
            filler_word_rate=0.0, tremor_score=0.0,
            transcript=None, flags=[reason],
        )
