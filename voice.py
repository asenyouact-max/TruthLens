"""
Voice prosody analyzer.
Extracts pitch, energy, speech rate, pauses, and filler words.
Uses Whisper for transcription + librosa for acoustic features.
"""
import numpy as np
import librosa
import whisper
from dataclasses import dataclass
from typing import Optional, List
import structlog
import re

log = structlog.get_logger()

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "sort of", "kind of", "basically", "literally", "actually"
}


@dataclass
class VoiceResult:
    pitch_elevation: float = 0.0
    speech_rate_change: float = 0.0
    filler_word_rate: float = 0.0
    pause_before_answer: float = 0.0
    voice_tremor: float = 0.0
    transcript: str = ""
    word_count: int = 0
    duration_seconds: float = 0.0


class VoiceDetector:
    def __init__(self, whisper_model: str = "base"):
        log.info("Loading Whisper model", model=whisper_model)
        self.whisper = whisper.load_model(whisper_model)
        self._baseline_pitch: Optional[float] = None
        self._baseline_rate: Optional[float] = None

    def set_baseline(self, pitch: float, rate: float):
        self._baseline_pitch = pitch
        self._baseline_rate = rate

    def analyze_audio(self, audio_path: str) -> VoiceResult:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr

        # Pitch (F0) via PYIN
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        mean_pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # Pitch elevation vs baseline
        if self._baseline_pitch and self._baseline_pitch > 0:
            pitch_elevation = float(np.clip(
                (mean_pitch - self._baseline_pitch) / self._baseline_pitch, -1, 1
            ))
        else:
            pitch_elevation = float(np.clip((pitch_std / (mean_pitch + 1e-6)) - 0.3, 0, 1))

        # Voice tremor — variance in F0 over time
        tremor = float(np.clip(pitch_std / (mean_pitch + 1e-6) * 2, 0, 1))

        # Transcription
        result = self.whisper.transcribe(audio_path, language="en")
        transcript = result["text"].strip()
        segments = result.get("segments", [])

        # Speech rate (words per minute)
        words = transcript.split()
        word_count = len(words)
        speech_rate = (word_count / duration) * 60 if duration > 0 else 0

        if self._baseline_rate and self._baseline_rate > 0:
            rate_change = abs(speech_rate - self._baseline_rate) / self._baseline_rate
        else:
            rate_change = 0.0

        # Filler word rate
        text_lower = transcript.lower()
        filler_count = sum(text_lower.count(f" {fw} ") for fw in FILLER_WORDS)
        sentence_count = max(len(re.split(r"[.!?]", transcript)), 1)
        filler_rate = float(np.clip(filler_count / sentence_count / 3, 0, 1))

        # Pauses — detect silence gaps from Whisper segments
        pauses = self._detect_pauses(segments)
        avg_pause = float(np.mean(pauses)) if pauses else 0.0
        pause_score = float(np.clip(avg_pause / 3.0, 0, 1))  # normalize to 3s max

        return VoiceResult(
            pitch_elevation=float(np.clip(abs(pitch_elevation), 0, 1)),
            speech_rate_change=float(np.clip(rate_change, 0, 1)),
            filler_word_rate=filler_rate,
            pause_before_answer=pause_score,
            voice_tremor=tremor,
            transcript=transcript,
            word_count=word_count,
            duration_seconds=duration,
        )

    def _detect_pauses(self, segments: List[dict]) -> List[float]:
        pauses = []
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i - 1]["end"]
            if gap > 0.5:  # pauses > 500ms
                pauses.append(gap)
        return pauses
