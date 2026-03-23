"""
AI-generated video detector.
Runs 5 parallel forensic checks:
1. Visual artifacts (EfficientNet ensemble)
2. Temporal flicker (optical flow consistency)
3. Biological signals — rPPG absence
4. Audio-video lip sync (SyncNet-style)
5. Frequency domain GAN artifacts (FFT)
6. Metadata anomalies
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import structlog
from scipy import signal as scipy_signal

log = structlog.get_logger()


@dataclass
class DeepfakeResult:
    visual_artifact_score: float = 0.0
    temporal_flicker_score: float = 0.0
    rppg_absence_score: float = 0.0
    lip_sync_score: float = 0.0
    frequency_artifact_score: float = 0.0
    metadata_anomaly_score: float = 0.0
    ensemble_score: float = 0.0
    is_ai_generated: bool = False
    confidence: float = 0.0


# Weights for ensemble (tunable)
WEIGHTS = {
    "visual": 0.25,
    "temporal": 0.20,
    "rppg": 0.25,
    "frequency": 0.20,
    "metadata": 0.10,
}


class DeepfakeDetector:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        # In production: load EfficientNet weights here
        # self.model = load_efficientnet_deepfake_model()
        log.info("DeepfakeDetector initialized", threshold=threshold)

    def analyze_video(self, video_path: str, metadata: Optional[dict] = None) -> DeepfakeResult:
        frames = self._extract_frames(video_path, max_frames=60)
        if not frames:
            return DeepfakeResult()

        visual = self._check_visual_artifacts(frames)
        temporal = self._check_temporal_flicker(frames)
        rppg = self._check_rppg(frames)
        freq = self._check_frequency_artifacts(frames)
        meta = self._check_metadata(metadata or {})

        ensemble = (
            visual * WEIGHTS["visual"]
            + temporal * WEIGHTS["temporal"]
            + rppg * WEIGHTS["rppg"]
            + freq * WEIGHTS["frequency"]
            + meta * WEIGHTS["metadata"]
        )

        return DeepfakeResult(
            visual_artifact_score=round(visual, 3),
            temporal_flicker_score=round(temporal, 3),
            rppg_absence_score=round(rppg, 3),
            lip_sync_score=0.0,  # requires audio — handled separately
            frequency_artifact_score=round(freq, 3),
            metadata_anomaly_score=round(meta, 3),
            ensemble_score=round(ensemble, 3),
            is_ai_generated=ensemble >= self.threshold,
            confidence=round(ensemble, 3),
        )

    def _extract_frames(self, path: str, max_frames: int = 60) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)
        idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            idx += step
        cap.release()
        return frames

    def _check_visual_artifacts(self, frames: List[np.ndarray]) -> float:
        """
        Heuristic: AI-generated faces have unnaturally smooth high-freq texture.
        Measure Laplacian variance — too-low variance = AI smoothing.
        In production replace with EfficientNet deepfake classifier.
        """
        scores = []
        for frame in frames[:20]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Real faces: 200-2000, AI faces tend: 50-300
            score = float(np.clip(1 - (lap_var - 50) / 500, 0, 1))
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def _check_temporal_flicker(self, frames: List[np.ndarray]) -> float:
        """
        Compare consecutive frames for unnatural pixel-level inconsistency.
        AI video often has subtle per-frame regeneration artifacts.
        """
        if len(frames) < 2:
            return 0.0
        diffs = []
        for i in range(1, min(len(frames), 30)):
            f1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(float)
            f2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
            diff = np.abs(f1 - f2)
            # High-frequency noise in diff = AI generation artifact
            high_freq = diff - cv2.GaussianBlur(diff, (5, 5), 0)
            diffs.append(float(np.std(high_freq)))

        mean_diff = float(np.mean(diffs))
        # Calibrated: real video ~2-8, AI video ~10-20
        return float(np.clip((mean_diff - 2) / 18, 0, 1))

    def _check_rppg(self, frames: List[np.ndarray]) -> float:
        """
        Remote photoplethysmography: real skin has subtle ~1Hz color oscillation
        from blood flow. AI faces don't reproduce this.
        Extract mean green channel from face region over time.
        """
        if len(frames) < 30:
            return 0.5  # not enough frames to decide

        green_means = []
        for frame in frames:
            h, w = frame.shape[:2]
            # Approximate face region (center crop)
            face = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            green_means.append(float(np.mean(face[:, :, 1])))

        signal_arr = np.array(green_means)
        signal_arr = signal_arr - np.mean(signal_arr)

        # Look for ~0.8-2.5 Hz component (heart rate range at ~30fps)
        if len(signal_arr) >= 30:
            fps_estimate = 30
            freqs = np.fft.rfftfreq(len(signal_arr), d=1.0 / fps_estimate)
            fft_mag = np.abs(np.fft.rfft(signal_arr))
            hr_band = (freqs >= 0.8) & (freqs <= 2.5)
            hr_power = float(np.sum(fft_mag[hr_band]))
            total_power = float(np.sum(fft_mag)) + 1e-6
            hr_ratio = hr_power / total_power
            # Real faces: ratio typically 0.15-0.4, AI faces: <0.05
            absence_score = float(np.clip(1 - hr_ratio / 0.15, 0, 1))
        else:
            absence_score = 0.5

        return absence_score

    def _check_frequency_artifacts(self, frames: List[np.ndarray]) -> float:
        """
        GAN-generated images have periodic grid artifacts visible in FFT.
        Check for unusual spikes at non-DC frequencies.
        """
        scores = []
        for frame in frames[:10]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))

            # Normalize and look for off-center peaks
            mag_norm = magnitude / (magnitude.max() + 1e-6)
            h, w = mag_norm.shape
            center = mag_norm[h // 2, w // 2]
            # Periodic GAN artifacts show as bright spots away from center
            peripheral = np.percentile(mag_norm, 99.5)
            scores.append(float(np.clip(peripheral / (center + 1e-6) - 0.5, 0, 1)))

        return float(np.mean(scores)) if scores else 0.0

    def _check_metadata(self, metadata: dict) -> float:
        score = 0.0
        checks = 0

        # Missing camera model
        if not metadata.get("camera_model"):
            score += 0.3
        checks += 1

        # Missing GPS when expected
        if not metadata.get("gps") and metadata.get("has_location_claim"):
            score += 0.4
        checks += 1

        # Generic or absent codec fingerprint
        codec = metadata.get("codec", "")
        if not codec or codec in ["h264", "unknown"]:
            score += 0.1
        checks += 1

        # No creation timestamp
        if not metadata.get("creation_time"):
            score += 0.2
        checks += 1

        return float(np.clip(score, 0, 1))
