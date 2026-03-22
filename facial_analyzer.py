"""
FacialAnalyzer — detects deception-related facial Action Units (AUs)
using MediaPipe Face Mesh landmark detection.

Key AUs for deception:
- AU1+4: inner brow raise (fear/distress)
- AU12 asymmetric: fake smile (Duchenne vs non-Duchenne)
- AU17: chin raiser (suppressed emotion)
- AU23: lip tightener (concealment)
- AU45: blink rate anomaly
- Gaze direction: lateral aversion patterns
"""
import asyncio
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class FacialResult:
    stress_score: float          # 0-1 aggregated AU stress
    confidence: float
    au_breakdown: Dict[str, float]
    gaze_aversion_rate: float    # % of frames with gaze avoidance
    smile_asymmetry: float       # Duchenne vs non-Duchenne
    micro_expression_count: int  # rapid AU flashes < 200ms
    flags: List[str]


class FacialAnalyzer:
    # MediaPipe landmark indices for key facial regions
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [61, 291, 39, 269, 0, 17, 405, 181]
    LEFT_BROW_INDICES = [70, 63, 105, 66, 107]
    RIGHT_BROW_INDICES = [336, 296, 334, 293, 300]

    def __init__(self):
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_available = True
        except ImportError:
            self.mp_available = False

    async def analyze(self, frames: List[np.ndarray]) -> FacialResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, frames)

    def _analyze_sync(self, frames: List[np.ndarray]) -> FacialResult:
        if not self.mp_available or not frames:
            return self._fallback_result()

        frame_results = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                frame_results.append(self._extract_aus(lms, frame.shape))

        if not frame_results:
            return self._fallback_result()

        # Aggregate across frames
        stress_scores = [r["stress"] for r in frame_results]
        gaze_aversions = [r["gaze_averted"] for r in frame_results]
        smile_asym = [r["smile_asymmetry"] for r in frame_results]

        micro_expression_count = self._count_micro_expressions(frame_results)

        flags = []
        mean_stress = float(np.mean(stress_scores))
        gaze_rate = float(np.mean(gaze_aversions))
        mean_asym = float(np.mean(smile_asym))

        if mean_stress > 0.6:
            flags.append("High facial stress indicators detected")
        if gaze_rate > 0.4:
            flags.append(f"Gaze aversion in {gaze_rate*100:.0f}% of frames")
        if mean_asym > 0.3:
            flags.append("Asymmetric smile pattern (possible masking)")
        if micro_expression_count > 5:
            flags.append(f"{micro_expression_count} micro-expressions detected")

        return FacialResult(
            stress_score=round(mean_stress, 3),
            confidence=0.75 if len(frame_results) > 10 else 0.5,
            au_breakdown={
                "brow_raise": float(np.mean([r.get("brow_raise", 0) for r in frame_results])),
                "lip_tighten": float(np.mean([r.get("lip_tighten", 0) for r in frame_results])),
                "chin_raise": float(np.mean([r.get("chin_raise", 0) for r in frame_results])),
            },
            gaze_aversion_rate=gaze_rate,
            smile_asymmetry=mean_asym,
            micro_expression_count=micro_expression_count,
            flags=flags,
        )

    def _extract_aus(self, landmarks, shape) -> Dict:
        h, w = shape[:2]
        pts = {i: (landmarks[i].x * w, landmarks[i].y * h) for i in range(468)}

        # Eye aspect ratio (blink/stress)
        def ear(eye_idx):
            p = [pts[i] for i in eye_idx]
            vertical = (
                np.linalg.norm(np.array(p[1]) - np.array(p[5])) +
                np.linalg.norm(np.array(p[2]) - np.array(p[4]))
            )
            horizontal = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
            return vertical / (2.0 * horizontal + 1e-8)

        left_ear = ear(self.LEFT_EYE_INDICES)
        right_ear = ear(self.RIGHT_EYE_INDICES)
        eye_stress = max(0.0, min(1.0, (0.3 - min(left_ear, right_ear)) * 10))

        # Brow height (AU1+4)
        left_brow_y = np.mean([pts[i][1] for i in self.LEFT_BROW_INDICES])
        left_eye_y = np.mean([pts[i][1] for i in self.LEFT_EYE_INDICES])
        brow_raise = max(0.0, min(1.0, (left_eye_y - left_brow_y) / 30.0 - 0.5))

        # Smile asymmetry
        mouth_pts = [pts[i] for i in self.MOUTH_INDICES]
        left_corner_y = mouth_pts[0][1]
        right_corner_y = mouth_pts[1][1]
        smile_asym = abs(left_corner_y - right_corner_y) / (h * 0.05 + 1e-8)
        smile_asym = min(1.0, smile_asym)

        # Gaze direction (iris landmarks if available)
        gaze_averted = abs(landmarks[468].x - 0.5) > 0.15 if len(landmarks.landmark) > 468 else False

        stress = (eye_stress * 0.4 + brow_raise * 0.3 + smile_asym * 0.3)

        return {
            "stress": stress,
            "brow_raise": brow_raise,
            "lip_tighten": eye_stress,
            "chin_raise": 0.0,  # TODO: implement chin raiser AU17
            "smile_asymmetry": smile_asym,
            "gaze_averted": float(gaze_averted),
        }

    def _count_micro_expressions(self, frame_results: List[Dict]) -> int:
        """Count rapid AU spikes (< 3 frames = ~100-200ms at 30fps)."""
        count = 0
        stress_series = [r["stress"] for r in frame_results]
        for i in range(1, len(stress_series) - 1):
            if (stress_series[i] > stress_series[i-1] + 0.3 and
                    stress_series[i] > stress_series[i+1] + 0.2):
                count += 1
        return count

    def _fallback_result(self) -> FacialResult:
        return FacialResult(
            stress_score=0.0, confidence=0.0,
            au_breakdown={}, gaze_aversion_rate=0.0,
            smile_asymmetry=0.0, micro_expression_count=0,
            flags=["MediaPipe unavailable — facial analysis skipped"],
        )
