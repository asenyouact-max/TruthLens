"""
DeepfakeDetector — runs 6 parallel forensic checks:
1. Visual artifact detection (EfficientNet/Xception ensemble)
2. Temporal consistency (frame-to-frame optical flow)
3. Biological signals (rPPG heartbeat, blink rate)
4. Audio-video sync (SyncNet-style lip-audio alignment)
5. Metadata forensics (EXIF, codec fingerprints)
6. Frequency domain (FFT GAN grid artifacts)
"""
import asyncio
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
from models.schemas import DeepfakeBreakdown


@dataclass
class DeepfakeResult:
    ensemble_score: float  # 0-1, higher = more likely AI
    visual_artifacts_score: float
    temporal_consistency_score: float
    biological_signals_score: float
    av_sync_score: float
    metadata_score: float
    frequency_artifacts_score: float
    flags: List[str]  # human-readable flags raised


class DeepfakeDetector:
    def __init__(self):
        self._load_models()

    def _load_models(self):
        """
        Load EfficientNet + Xception fine-tuned on FaceForensics++.
        In production: load from S3 or local model cache.
        Using placeholder structure — replace with actual model weights.
        """
        # TODO: load actual model weights
        # self.efficientnet = load_model("models/efficientnet_ff++.h5")
        # self.xception = load_model("models/xception_ff++.h5")
        self.models_loaded = False  # set True when weights are loaded

    async def analyze(
        self, frames: List[np.ndarray], audio_path: str, video_path: str
    ) -> DeepfakeBreakdown:
        loop = asyncio.get_event_loop()

        # Run all 6 detectors in parallel thread pool
        visual_task = loop.run_in_executor(None, self._visual_artifacts, frames)
        temporal_task = loop.run_in_executor(None, self._temporal_consistency, frames)
        bio_task = loop.run_in_executor(None, self._biological_signals, frames)
        av_task = loop.run_in_executor(None, self._av_sync, frames, audio_path)
        meta_task = loop.run_in_executor(None, self._metadata_forensics, video_path)
        freq_task = loop.run_in_executor(None, self._frequency_analysis, frames)

        (visual, temporal, bio, av_sync, meta, freq) = await asyncio.gather(
            visual_task, temporal_task, bio_task, av_task, meta_task, freq_task
        )

        # Weighted ensemble — bio signals + visual artifacts weighted highest
        ensemble = (
            visual * 0.25
            + temporal * 0.15
            + bio * 0.25
            + av_sync * 0.15
            + meta * 0.10
            + freq * 0.10
        )

        return DeepfakeBreakdown(
            visual_artifacts_score=round(visual, 3),
            temporal_consistency_score=round(temporal, 3),
            biological_signals_score=round(bio, 3),
            av_sync_score=round(av_sync, 3),
            metadata_score=round(meta, 3),
            frequency_artifacts_score=round(freq, 3),
            ensemble_score=round(ensemble, 3),
        )

    def _visual_artifacts(self, frames: List[np.ndarray]) -> float:
        """
        Run EfficientNet + Xception on face crops.
        Returns mean probability of being AI-generated.
        """
        if not self.models_loaded or not frames:
            return self._heuristic_visual(frames)

        scores = []
        for frame in frames[::3]:  # sample every 3rd frame for speed
            face = self._crop_face(frame)
            if face is None:
                continue
            face_resized = cv2.resize(face, (224, 224)) / 255.0
            face_batch = np.expand_dims(face_resized, 0)
            # score = (self.efficientnet.predict(face_batch)[0][0] +
            #          self.xception.predict(face_batch)[0][0]) / 2
            scores.append(0.5)  # placeholder
        return float(np.mean(scores)) if scores else 0.5

    def _heuristic_visual(self, frames: List[np.ndarray]) -> float:
        """
        Heuristic fallback: checks for over-smooth skin texture (low local variance),
        unnatural color histograms, and edge coherence.
        """
        scores = []
        for frame in frames[::5]:
            face = self._crop_face(frame)
            if face is None:
                continue
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Low local variance = suspiciously smooth (AI skin)
            local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize: real faces typically 200-800, AI faces <100
            smoothness_score = max(0.0, min(1.0, 1.0 - (local_var / 400.0)))
            scores.append(smoothness_score)
        return float(np.mean(scores)) if scores else 0.3

    def _temporal_consistency(self, frames: List[np.ndarray]) -> float:
        """
        Optical flow between consecutive frames.
        High inter-frame noise on static regions = AI artifact.
        """
        if len(frames) < 2:
            return 0.3
        anomaly_scores = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for frame in frames[1::2]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            # High variance in regions that should be still = anomaly
            anomaly_scores.append(float(np.std(magnitude)))
            prev_gray = curr_gray
        normalized = min(1.0, np.mean(anomaly_scores) / 5.0)
        return float(normalized)

    def _biological_signals(self, frames: List[np.ndarray]) -> float:
        """
        rPPG: detect heartbeat signal from skin color oscillation.
        Blink rate: count eye closures per second.
        AI videos rarely reproduce these correctly.
        """
        rppg_score = self._rppg_check(frames)
        blink_score = self._blink_rate_check(frames)
        return (rppg_score * 0.6 + blink_score * 0.4)

    def _rppg_check(self, frames: List[np.ndarray]) -> float:
        """
        Extract mean green channel from forehead ROI over time.
        Real faces show ~60-100 BPM oscillation. AI faces show noise.
        """
        green_signals = []
        for frame in frames:
            face = self._crop_face(frame)
            if face is None:
                continue
            h, w = face.shape[:2]
            forehead = face[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)]
            if forehead.size == 0:
                continue
            green_signals.append(float(np.mean(forehead[:, :, 1])))

        if len(green_signals) < 10:
            return 0.4  # not enough data

        # Check for periodic signal in 0.8-2.5 Hz range (48-150 BPM)
        signal = np.array(green_signals) - np.mean(green_signals)
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0/3)  # 3 fps sample rate
        heart_band = (freqs >= 0.8) & (freqs <= 2.5)
        heart_power = np.sum(fft[heart_band])
        total_power = np.sum(fft) + 1e-8
        heartbeat_ratio = heart_power / total_power
        # Real face: ratio ~0.3-0.6. AI face: ratio <0.1
        return max(0.0, min(1.0, 1.0 - (heartbeat_ratio / 0.3)))

    def _blink_rate_check(self, frames: List[np.ndarray]) -> float:
        """
        Count blink events. Humans blink 12-20 times/min.
        AI videos often have 0 blinks or robotic regular blinking.
        """
        # Simplified: use eye aspect ratio from face landmarks
        # TODO: integrate dlib or MediaPipe for actual landmark detection
        return 0.35  # placeholder — replace with real blink detection

    def _av_sync(self, frames: List[np.ndarray], audio_path: str) -> float:
        """
        Lip-audio alignment check.
        Measure mouth openness from frames, compare to audio energy envelope.
        Mismatch indicates deepfake lip sync.
        """
        # TODO: replace with SyncNet or active speaker detection model
        return 0.3  # placeholder

    def _metadata_forensics(self, video_path: str) -> float:
        """
        Check for missing/generic metadata that real cameras always produce.
        AI-generated video: missing GPS, generic codec, no camera model.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", video_path],
                capture_output=True, text=True, timeout=10
            )
            import json
            meta = json.loads(result.stdout)
            suspicious_score = 0.0
            tags = meta.get("format", {}).get("tags", {})
            if not tags.get("make") and not tags.get("Make"):
                suspicious_score += 0.2  # no camera make
            if not tags.get("creation_time"):
                suspicious_score += 0.2  # no creation time
            if not tags.get("location") and not tags.get("com.apple.quicktime.location.ISO6709"):
                suspicious_score += 0.1  # no GPS (soft signal only)
            return min(1.0, suspicious_score)
        except Exception:
            return 0.3

    def _frequency_analysis(self, frames: List[np.ndarray]) -> float:
        """
        FFT on face crops. GAN-generated images leave periodic grid artifacts
        visible as peaks in frequency domain.
        """
        scores = []
        for frame in frames[::5]:
            face = self._crop_face(frame)
            if face is None:
                continue
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)
            magnitude = 20 * np.log(np.abs(dft_shift) + 1)
            # GAN artifacts: unusually high peaks at non-center frequencies
            center_y, center_x = magnitude.shape[0]//2, magnitude.shape[1]//2
            outer = magnitude.copy()
            outer[center_y-10:center_y+10, center_x-10:center_x+10] = 0
            peak_score = float(np.percentile(outer, 99.5)) / 200.0
            scores.append(min(1.0, peak_score))
        return float(np.mean(scores)) if scores else 0.3

    def _crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Simple Haar cascade face crop — replace with MTCNN for production."""
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return frame[y:y+h, x:x+w]
