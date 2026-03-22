"""
PoseAnalyzer — detects deception-related body language signals:
- Self-touching (face/neck) — high deception indicator
- Postural shifts and fidgeting
- Arm barrier / closed posture
- Gesture-speech asynchrony
- Head movement patterns
"""
import asyncio
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class PoseResult:
    deception_score: float
    confidence: float
    self_touch_rate: float       # % frames with hand-to-face/neck contact
    posture_shift_count: int     # significant posture changes
    closed_posture_rate: float   # % frames with arm barrier
    head_movement_variance: float
    flags: List[str]


class PoseAnalyzer:
    def __init__(self):
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_available = True
        except ImportError:
            self.mp_available = False

    async def analyze(self, frames: List[np.ndarray]) -> PoseResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, frames)

    def _analyze_sync(self, frames: List[np.ndarray]) -> PoseResult:
        if not self.mp_available or not frames:
            return self._fallback_result()

        frame_data = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.holistic.process(rgb)
            if result.pose_landmarks and result.left_hand_landmarks:
                data = self._extract_signals(result, frame.shape)
                frame_data.append(data)

        if not frame_data:
            return self._fallback_result()

        self_touch_rate = float(np.mean([d["self_touch"] for d in frame_data]))
        closed_posture_rate = float(np.mean([d["closed_posture"] for d in frame_data]))

        head_positions = [d["head_y"] for d in frame_data if d["head_y"] is not None]
        head_variance = float(np.var(head_positions)) if head_positions else 0.0

        posture_shifts = self._count_posture_shifts(frame_data)

        # Deception score: weighted combination
        deception_score = (
            self_touch_rate * 0.40
            + closed_posture_rate * 0.25
            + min(1.0, posture_shifts / 5.0) * 0.20
            + min(1.0, head_variance / 100.0) * 0.15
        )

        flags = []
        if self_touch_rate > 0.3:
            flags.append(f"Self-touching detected in {self_touch_rate*100:.0f}% of frames")
        if closed_posture_rate > 0.5:
            flags.append("Closed/defensive posture throughout")
        if posture_shifts > 4:
            flags.append(f"{posture_shifts} significant posture shifts detected")
        if head_variance > 150:
            flags.append("Excessive head movement / fidgeting")

        return PoseResult(
            deception_score=round(deception_score, 3),
            confidence=0.70 if len(frame_data) > 10 else 0.45,
            self_touch_rate=round(self_touch_rate, 3),
            posture_shift_count=posture_shifts,
            closed_posture_rate=round(closed_posture_rate, 3),
            head_movement_variance=round(head_variance, 3),
            flags=flags,
        )

    def _extract_signals(self, result, shape) -> Dict:
        h, w = shape[:2]

        # Pose landmarks
        pose = result.pose_landmarks.landmark

        # Head position (nose landmark = 0)
        head_y = pose[0].y * h

        # Shoulder midpoint for normalization
        left_shoulder = np.array([pose[11].x * w, pose[11].y * h])
        right_shoulder = np.array([pose[12].x * w, pose[12].y * h])
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        # Closed posture: wrists crossed in front of body
        left_wrist = np.array([pose[15].x * w, pose[15].y * h])
        right_wrist = np.array([pose[16].x * w, pose[16].y * h])
        wrist_dist = np.linalg.norm(left_wrist - right_wrist)
        closed_posture = float(wrist_dist < shoulder_width * 0.5)

        # Self-touch: hand landmark near face landmarks
        self_touch = 0.0
        face_y_region = (pose[0].y * h - 80, pose[0].y * h + 120)  # face bbox estimate
        for hand_lm in [result.left_hand_landmarks, result.right_hand_landmarks]:
            if hand_lm:
                hand_y_positions = [lm.y * h for lm in hand_lm.landmark]
                hand_mean_y = np.mean(hand_y_positions)
                if face_y_region[0] < hand_mean_y < face_y_region[1]:
                    self_touch = 1.0
                    break

        return {
            "head_y": head_y,
            "closed_posture": closed_posture,
            "self_touch": self_touch,
            "left_shoulder_x": left_shoulder[0],
        }

    def _count_posture_shifts(self, frame_data: List[Dict]) -> int:
        """Count significant shoulder-level position changes."""
        if len(frame_data) < 3:
            return 0
        positions = [d["left_shoulder_x"] for d in frame_data]
        shifts = 0
        for i in range(1, len(positions)):
            if abs(positions[i] - positions[i-1]) > 20:  # pixels
                shifts += 1
        return shifts

    def _fallback_result(self) -> PoseResult:
        return PoseResult(
            deception_score=0.0, confidence=0.0,
            self_touch_rate=0.0, posture_shift_count=0,
            closed_posture_rate=0.0, head_movement_variance=0.0,
            flags=["MediaPipe unavailable — pose analysis skipped"],
        )
