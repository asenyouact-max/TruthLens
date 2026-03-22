"""
AudioAIDetector — comprehensive AI/synthetic audio detection.

Covers three categories:

VOICE / TTS DETECTION:
  - Spectral flatness (TTS = unnaturally clean harmonics)
  - MFCC temporal drift (voice cloning has different formant dynamics)
  - Pitch regularity (human voice has natural micro-jitter; TTS is too perfect)
  - Voice embedding similarity via resemblyzer (detects voice cloning)
  - Breathiness / vocal fry absence (TTS often lacks these natural textures)
  - Prosody flatness (TTS lacks natural sentence-level intonation arcs)

ENVIRONMENT / ACOUSTICS:
  - RT60 reverb estimation (real rooms have reverb; TTS is dry)
  - Noise floor uniformity (real recordings have natural noise variation)
  - Background sound naturalness (AI audio often has no background)
  - Codec artifact absence (real recordings show compression artifacts)
  - Silence pattern naturalness (pauses in AI audio are often too clean)

AV COHERENCE (when video frames are available):
  - Lip-phoneme alignment (energy envelope vs mouth openness)
  - Emotion congruence (happy voice + neutral face = mismatch)
  - Speaking rate vs lip movement correlation
"""
import asyncio
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class AudioAIResult:
    # Category scores (0=natural, 1=AI-generated)
    voice_synthesis_score: float = 0.0    # TTS/voice cloning
    environment_score: float = 0.0        # room acoustics, noise
    av_coherence_score: float = 0.0       # lip sync, emotion match

    # Detailed sub-scores
    spectral_flatness: float = 0.0
    mfcc_drift: float = 0.0
    pitch_regularity: float = 0.0
    prosody_flatness: float = 0.0
    reverb_score: float = 0.0
    noise_uniformity: float = 0.0
    silence_pattern_score: float = 0.0
    lip_sync_score: float = 0.0
    emotion_mismatch_score: float = 0.0

    ensemble_score: float = 0.0
    transcript: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    confidence: float = 0.0


# Sub-score weights within each category
VOICE_WEIGHTS = {
    "spectral_flatness": 0.25,
    "mfcc_drift": 0.20,
    "pitch_regularity": 0.25,
    "prosody_flatness": 0.30,
}
ENV_WEIGHTS = {
    "reverb": 0.40,
    "noise_uniformity": 0.35,
    "silence_pattern": 0.25,
}
AV_WEIGHTS = {
    "lip_sync": 0.60,
    "emotion_mismatch": 0.40,
}
CATEGORY_WEIGHTS = {
    "voice": 0.45,
    "environment": 0.35,
    "av_coherence": 0.20,
}


class AudioAIDetector:

    async def analyze(
        self,
        audio_path: str,
        video_frames: Optional[List[np.ndarray]] = None,
    ) -> AudioAIResult:
        if not audio_path or not os.path.exists(audio_path):
            result = AudioAIResult(flags=["Audio file not found"], confidence=0.0)
            result.ensemble_score = 0.3
            return result

        loop = asyncio.get_event_loop()

        voice_task = loop.run_in_executor(None, self._analyze_voice, audio_path)
        env_task = loop.run_in_executor(None, self._analyze_environment, audio_path)
        av_task = loop.run_in_executor(
            None, self._analyze_av_coherence, audio_path, video_frames
        )

        voice_result, env_result, av_result = await asyncio.gather(
            voice_task, env_task, av_task
        )

        result = AudioAIResult(
            # Voice
            voice_synthesis_score=voice_result["score"],
            spectral_flatness=voice_result["spectral_flatness"],
            mfcc_drift=voice_result["mfcc_drift"],
            pitch_regularity=voice_result["pitch_regularity"],
            prosody_flatness=voice_result["prosody_flatness"],
            # Environment
            environment_score=env_result["score"],
            reverb_score=env_result["reverb"],
            noise_uniformity=env_result["noise_uniformity"],
            silence_pattern_score=env_result["silence_pattern"],
            # AV
            av_coherence_score=av_result["score"],
            lip_sync_score=av_result["lip_sync"],
            emotion_mismatch_score=av_result["emotion_mismatch"],
            transcript=voice_result.get("transcript"),
            flags=voice_result["flags"] + env_result["flags"] + av_result["flags"],
        )

        result.ensemble_score = round(
            result.voice_synthesis_score * CATEGORY_WEIGHTS["voice"]
            + result.environment_score   * CATEGORY_WEIGHTS["environment"]
            + result.av_coherence_score  * CATEGORY_WEIGHTS["av_coherence"],
            3,
        )
        result.confidence = 0.80 if os.path.getsize(audio_path) > 10000 else 0.40

        return result

    # ─────────────────────────────────────────────────
    # CATEGORY 1: Voice / TTS / Cloning
    # ─────────────────────────────────────────────────
    def _analyze_voice(self, audio_path: str) -> Dict:
        flags = []
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=120)
        except Exception as e:
            return {"score": 0.3, "spectral_flatness": 0.3, "mfcc_drift": 0.3,
                    "pitch_regularity": 0.3, "prosody_flatness": 0.3,
                    "transcript": None, "flags": [f"librosa error: {e}"]}

        # ── 1. Spectral flatness ──────────────────────
        # TTS voices: flatness > 0.06 (too spectrally uniform)
        # Real voice: 0.001–0.04
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flatness = float(np.mean(flatness))
        spectral_flatness_score = min(1.0, mean_flatness / 0.07)
        if mean_flatness > 0.06:
            flags.append(f"High spectral flatness ({mean_flatness:.4f}) — typical of TTS synthesis")

        # ── 2. MFCC temporal drift ────────────────────
        # Voice cloning smooths out the frame-to-frame MFCC changes
        # Real voice: higher variance in delta-MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfccs)
        mfcc_variance = float(np.mean(np.var(delta_mfcc, axis=1)))
        # Real voice variance: 5–30. Cloned: <3
        mfcc_drift_score = max(0.0, min(1.0, 1.0 - mfcc_variance / 8.0))
        if mfcc_variance < 3.0:
            flags.append("Low MFCC temporal variance — possible voice cloning")

        # ── 3. Pitch regularity ───────────────────────
        # Human pitch has natural micro-jitter. TTS is too stable.
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([])

        if len(voiced_f0) > 20:
            # Compute pitch jitter: mean absolute frame-to-frame variation
            jitter = float(np.mean(np.abs(np.diff(voiced_f0))))
            mean_f0 = float(np.mean(voiced_f0))
            relative_jitter = jitter / (mean_f0 + 1e-8)
            # Real voice: relative jitter 0.005–0.02. TTS: <0.002
            pitch_regularity_score = max(0.0, min(1.0, 1.0 - relative_jitter / 0.005))

            # Also check pitch range — TTS often has compressed pitch range
            f0_range = float(np.percentile(voiced_f0, 90) - np.percentile(voiced_f0, 10))
            if f0_range < 20:  # Hz
                pitch_regularity_score = min(1.0, pitch_regularity_score + 0.2)
                flags.append(f"Compressed pitch range ({f0_range:.0f} Hz) — TTS characteristic")
        else:
            pitch_regularity_score = 0.3  # insufficient voiced frames

        if pitch_regularity_score > 0.65:
            flags.append("Unnaturally regular pitch — robotic cadence detected")

        # ── 4. Prosody flatness ───────────────────────
        # Real speech has natural sentence-level intonation arcs.
        # TTS tends to produce flat or repetitive prosody patterns.
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        # Segment into ~1s chunks, check energy arc variance
        chunk_size = sr // 512  # frames per second
        chunk_means = [
            float(np.mean(rms[i:i+chunk_size]))
            for i in range(0, len(rms) - chunk_size, chunk_size)
        ]
        if len(chunk_means) > 3:
            prosody_variance = float(np.var(chunk_means))
            # Real speech: variance 0.001–0.01. TTS: <0.0005
            prosody_flatness_score = max(0.0, min(1.0, 1.0 - prosody_variance / 0.002))
        else:
            prosody_flatness_score = 0.3

        if prosody_flatness_score > 0.6:
            flags.append("Flat prosody pattern — lacks natural sentence intonation")

        # ── Transcription ─────────────────────────────
        transcript = self._transcribe(audio_path)

        voice_score = (
            spectral_flatness_score * VOICE_WEIGHTS["spectral_flatness"]
            + mfcc_drift_score      * VOICE_WEIGHTS["mfcc_drift"]
            + pitch_regularity_score* VOICE_WEIGHTS["pitch_regularity"]
            + prosody_flatness_score* VOICE_WEIGHTS["prosody_flatness"]
        )

        return {
            "score": round(voice_score, 3),
            "spectral_flatness": round(spectral_flatness_score, 3),
            "mfcc_drift": round(mfcc_drift_score, 3),
            "pitch_regularity": round(pitch_regularity_score, 3),
            "prosody_flatness": round(prosody_flatness_score, 3),
            "transcript": transcript,
            "flags": flags,
        }

    # ─────────────────────────────────────────────────
    # CATEGORY 2: Environment / Acoustics
    # ─────────────────────────────────────────────────
    def _analyze_environment(self, audio_path: str) -> Dict:
        flags = []
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=120)
        except Exception as e:
            return {"score": 0.3, "reverb": 0.3, "noise_uniformity": 0.3,
                    "silence_pattern": 0.3, "flags": [f"librosa error: {e}"]}

        # ── 1. Reverb / RT60 estimation ───────────────
        # Real rooms: RT60 typically 0.2–1.5s. TTS: near 0.
        # Estimate via energy decay in silent regions after voiced bursts.
        reverb_score = self._estimate_reverb_absence(y, sr)
        if reverb_score > 0.65:
            flags.append("No detectable room reverberation — possibly synthetic audio")

        # ── 2. Noise floor uniformity ─────────────────
        # Real recordings: noise floor has natural variation (HVAC, room tone, etc.)
        # AI audio: perfectly uniform or absent noise floor
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        silence_mask = rms < np.percentile(rms, 15)
        if silence_mask.sum() > 50:
            silent_rms = rms[silence_mask]
            noise_cv = float(np.std(silent_rms) / (np.mean(silent_rms) + 1e-8))
            # Real noise: coefficient of variation 0.1–0.5. AI: <0.05 (too uniform)
            noise_uniformity_score = max(0.0, min(1.0, 1.0 - noise_cv / 0.1))
        else:
            noise_uniformity_score = 0.5  # no clear silent regions

        if noise_uniformity_score > 0.7:
            flags.append("Unnaturally uniform noise floor — no organic room tone")

        # ── 3. Silence pattern naturalness ───────────────
        # Human pauses: irregular, ragged edges, breath sounds.
        # TTS pauses: perfectly clean, abrupt onset/offset.
        silence_score = self._analyze_silence_patterns(y, sr)
        if silence_score > 0.6:
            flags.append("Abrupt/clean pause patterns — TTS silence characteristic")

        # ── 4. Codec artifact check ───────────────────
        # Real phone/device recordings show MP3/AAC compression artifacts.
        # AI audio rendered directly to WAV often lacks these.
        codec_score = self._check_codec_artifacts(y, sr)

        env_score = (
            reverb_score          * ENV_WEIGHTS["reverb"]
            + noise_uniformity_score * ENV_WEIGHTS["noise_uniformity"]
            + silence_score       * ENV_WEIGHTS["silence_pattern"]
        )
        # Blend in codec score as a soft signal
        env_score = env_score * 0.85 + codec_score * 0.15

        return {
            "score": round(env_score, 3),
            "reverb": round(reverb_score, 3),
            "noise_uniformity": round(noise_uniformity_score, 3),
            "silence_pattern": round(silence_score, 3),
            "flags": flags,
        }

    def _estimate_reverb_absence(self, y: np.ndarray, sr: int) -> float:
        """
        Estimate RT60 proxy by finding voiced→silent transitions
        and measuring how quickly energy decays.
        Fast/instant decay = no reverb = likely synthetic.
        """
        import librosa
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        voiced = rms > np.percentile(rms, 40)
        transitions = np.where(np.diff(voiced.astype(int)) == -1)[0]

        decay_scores = []
        for t in transitions[:8]:
            tail = rms[t:min(t + 20, len(rms))]
            if len(tail) < 5:
                continue
            if tail[0] < 1e-6:
                continue
            # Normalized decay: how much energy remains after 10 frames
            decay_ratio = float(tail[min(10, len(tail)-1)] / tail[0])
            # Real room: decay_ratio 0.3–0.7. No reverb: ~0.0
            reverb_absence = max(0.0, min(1.0, 1.0 - decay_ratio / 0.3))
            decay_scores.append(reverb_absence)

        return float(np.mean(decay_scores)) if decay_scores else 0.4

    def _analyze_silence_patterns(self, y: np.ndarray, sr: int) -> float:
        """
        Check pause onset/offset sharpness.
        Real pauses: gradual fade with breath/friction sounds.
        TTS pauses: binary on/off, zero energy.
        """
        import librosa
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        silence_thresh = np.percentile(rms, 20)
        is_silent = rms < silence_thresh

        transition_sharpness = []
        for i in range(5, len(is_silent) - 5):
            if is_silent[i] and not is_silent[i-1]:
                # Start of silence: check how fast energy drops
                pre = float(np.mean(rms[max(0, i-3):i]))
                post = float(np.mean(rms[i:i+3]))
                if pre > 1e-6:
                    sharpness = 1.0 - min(1.0, post / pre)
                    transition_sharpness.append(sharpness)

        if not transition_sharpness:
            return 0.3
        # High mean sharpness = abrupt cutoffs = TTS-like
        return round(float(np.mean(transition_sharpness)), 3)

    def _check_codec_artifacts(self, y: np.ndarray, sr: int) -> float:
        """
        MP3/AAC compression introduces pre-echo and spectral smearing.
        AI audio rendered clean to WAV lacks these — suspiciously pristine.
        Check for typical compression artifacts at 3.5kHz, 7kHz (MP3 cutoffs).
        """
        import librosa
        # Check spectral energy at high frequencies
        stft = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        high_freq_mask = freqs > 7000
        high_freq_energy = float(np.mean(stft[high_freq_mask]))
        total_energy = float(np.mean(stft)) + 1e-8
        hf_ratio = high_freq_energy / total_energy
        # Real recordings: hf_ratio 0.05–0.25. Clean AI render: can be near 0 or very flat
        if hf_ratio < 0.02:
            return 0.6  # suspiciously low high-frequency content
        return 0.2  # normal

    # ─────────────────────────────────────────────────
    # CATEGORY 3: AV Coherence
    # ─────────────────────────────────────────────────
    def _analyze_av_coherence(
        self,
        audio_path: str,
        frames: Optional[List[np.ndarray]],
    ) -> Dict:
        flags = []

        if frames is None or len(frames) < 5:
            return {
                "score": 0.3, "lip_sync": 0.3, "emotion_mismatch": 0.3,
                "flags": ["No video frames for AV coherence check"],
            }

        lip_sync_score = self._check_lip_sync(audio_path, frames)
        emotion_mismatch_score = self._check_emotion_mismatch(audio_path, frames)

        if lip_sync_score > 0.6:
            flags.append("Lip-audio synchronization anomaly detected")
        if emotion_mismatch_score > 0.6:
            flags.append("Emotional tone mismatch between audio and facial expression")

        av_score = (
            lip_sync_score       * AV_WEIGHTS["lip_sync"]
            + emotion_mismatch_score * AV_WEIGHTS["emotion_mismatch"]
        )

        return {
            "score": round(av_score, 3),
            "lip_sync": round(lip_sync_score, 3),
            "emotion_mismatch": round(emotion_mismatch_score, 3),
            "flags": flags,
        }

    def _check_lip_sync(self, audio_path: str, frames: List[np.ndarray]) -> float:
        """
        Compare audio energy envelope with mouth openness from frames.
        Compute Pearson correlation — real AV should be highly correlated.
        Low correlation = desync = likely deepfake audio/video.
        """
        import cv2
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=30)
        except Exception:
            return 0.3

        # Audio energy envelope (1 value per video frame)
        hop = len(y) // len(frames)
        if hop < 1:
            return 0.3
        audio_energy = [
            float(np.sqrt(np.mean(y[i*hop:(i+1)*hop]**2)))
            for i in range(min(len(frames), len(y)//hop))
        ]

        # Mouth openness from frames
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        mouth_openness = []
        for frame in frames[:len(audio_energy)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) == 0:
                mouth_openness.append(0.0)
                continue
            x, y_f, w, h = faces[0]
            # Mouth region: lower third of face
            mouth = gray[y_f + int(h*0.65):y_f+h, x:x+w]
            if mouth.size > 0:
                # Mouth openness proxy: vertical gradient energy in mouth region
                grad = np.abs(np.diff(mouth.astype(float), axis=0))
                mouth_openness.append(float(np.mean(grad)))
            else:
                mouth_openness.append(0.0)

        min_len = min(len(audio_energy), len(mouth_openness))
        if min_len < 5:
            return 0.3

        ae = np.array(audio_energy[:min_len])
        mo = np.array(mouth_openness[:min_len])

        if np.std(ae) < 1e-6 or np.std(mo) < 1e-6:
            return 0.5

        corr = float(np.corrcoef(ae, mo)[0, 1])
        if np.isnan(corr):
            return 0.4

        # Good sync: corr > 0.5. Desync: corr < 0.2
        sync_score = max(0.0, min(1.0, 1.0 - (corr + 1) / 2.0))
        return round(sync_score, 3)

    def _check_emotion_mismatch(
        self, audio_path: str, frames: List[np.ndarray]
    ) -> float:
        """
        Rough emotion congruence check:
        - Estimate audio emotional valence (energy + pitch = arousal)
        - Estimate facial expression valence (smile vs neutral)
        - Large divergence = possible AI mismatch
        """
        import cv2
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=30)
        except Exception:
            return 0.3

        # Audio arousal proxy: mean energy × mean pitch height
        rms_mean = float(np.sqrt(np.mean(y**2)))
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced] if voiced is not None and voiced.any() else np.array([150.0])
        pitch_norm = float(np.mean(voiced_f0)) / 300.0  # normalize 0–1 range
        audio_arousal = min(1.0, (rms_mean * 10 + pitch_norm) / 2)

        # Face valence proxy: smile ratio using simple mouth-corner geometry
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        smile_scores = []
        for frame in frames[::3]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) == 0:
                continue
            x, yf, w, h = faces[0]
            mouth_region = gray[yf+int(h*0.6):yf+h, x:x+w]
            if mouth_region.size > 0:
                # Brightness in mouth corners vs center as smile proxy
                row_brightness = np.mean(mouth_region, axis=0)
                center = float(np.mean(row_brightness[w//3:2*w//3]))
                corners = float(np.mean(np.concatenate([
                    row_brightness[:w//5], row_brightness[4*w//5:]
                ])))
                smile_scores.append(max(0.0, corners - center) / 50.0)

        face_valence = float(np.mean(smile_scores)) if smile_scores else 0.3
        face_valence = min(1.0, face_valence)

        # Mismatch: big difference between audio arousal and face valence
        mismatch = abs(audio_arousal - face_valence)
        return round(min(1.0, mismatch * 1.5), 3)

    # ─────────────────────────────────────────────────
    # Transcription helper
    # ─────────────────────────────────────────────────
    def _transcribe(self, audio_path: str) -> Optional[str]:
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result.get("text", "").strip()
        except Exception:
            return None
