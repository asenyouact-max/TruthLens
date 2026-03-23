"""
Facial Action Unit (AU) + micro-expression detector.
Uses MediaPipe Face Mesh to extract 478 landmarks,
then maps them to FACS Action Units relevant to deception.
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import structlog

log = structlog.get_logger()

mp_face_mesh = mp.solutions.face_mesh

# Landmark index groups for key AUs
AU_LANDMARKS = {
    "AU1_inner_brow_raise": [107, 336],       # inner eyebrows
    "AU2_outer_brow_raise": [70, 300],         # outer eyebrows
    "AU4_brow_lowerer":     [107, 336, 9],     # glabella compression
    "AU6_cheek_raise":      [187, 411],        # cheek apples (Duchenne)
    "AU7_lid_tightener":    [159, 386],        # upper eyelid
    "AU12_lip_corner_pull": [61, 291],         # lip corners (smile)
    "AU17_chin_raiser":     [175],             # chin
    "AU20_lip_stretcher":   [61, 291, 17],     # horizontal lip stretch
    "AU23_lip_tightener":   [13, 14],          # lip contact
    "AU24_lip_pressor":     [13, 14, 17],      # lip compression
}


@dataclass
class FacialResult:
    micro_expression_score: float = 0.0
    gaze_aversion_score: float = 0.0
    blink_irregularity: float = 0.0
    lip_compression: float = 0.0
    asymmetric_smile: float = 0.0
    raw_aus: Dict[str, float] = field(default_factory=dict)


class FacialDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._blink_history: List[float] = []
        self._frame_count = 0
        log.info("FacialDetector initialized")

    def process_frame(self, frame: np.ndarray) -> Optional[FacialResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        pts = {i: (lm.x * w, lm.y * h, lm.z) for i, lm in enumerate(landmarks)}

        aus = self._compute_aus(pts, h, w)
        gaze = self._compute_gaze_aversion(pts)
        blink = self._compute_blink(pts)
        self._blink_history.append(blink)
        if len(self._blink_history) > 300:
            self._blink_history.pop(0)

        lip_compression = (aus.get("AU23_lip_tightener", 0) + aus.get("AU24_lip_pressor", 0)) / 2
        smile_left = abs(pts[61][1] - pts[291][1]) if 61 in pts and 291 in pts else 0
        smile_right = abs(pts[61][1] - pts[291][1]) if 61 in pts and 291 in pts else 0
        asymmetric_smile = abs(smile_left - smile_right) / (h * 0.01 + 1e-6)
        asymmetric_smile = min(asymmetric_smile, 1.0)

        micro_exp = self._micro_expression_score(aus)
        blink_irr = self._blink_irregularity()

        return FacialResult(
            micro_expression_score=micro_exp,
            gaze_aversion_score=gaze,
            blink_irregularity=blink_irr,
            lip_compression=float(np.clip(lip_compression, 0, 1)),
            asymmetric_smile=float(asymmetric_smile),
            raw_aus=aus,
        )

    def _compute_aus(self, pts: dict, h: int, w: int) -> Dict[str, float]:
        aus = {}
        # AU4: brow lowerer — distance between brows and eyes
        if all(k in pts for k in [107, 336, 159, 386]):
            brow_y = (pts[107][1] + pts[336][1]) / 2
            eye_y = (pts[159][1] + pts[386][1]) / 2
            aus["AU4_brow_lowerer"] = float(np.clip(1 - (eye_y - brow_y) / (h * 0.08), 0, 1))

        # AU24: lip compression — vertical distance between lips
        if all(k in pts for k in [13, 14]):
            lip_gap = abs(pts[13][1] - pts[14][1])
            aus["AU24_lip_pressor"] = float(np.clip(1 - lip_gap / (h * 0.03), 0, 1))

        # AU12: lip corner pull (smile)
        if all(k in pts for k in [61, 291, 13]):
            mouth_w = abs(pts[61][0] - pts[291][0])
            aus["AU12_lip_corner_pull"] = float(np.clip(mouth_w / (w * 0.25), 0, 1))

        return aus

    def _compute_gaze_aversion(self, pts: dict) -> float:
        # Iris position relative to eye corners — deviation from center = aversion
        if not all(k in pts for k in [468, 473, 133, 362]):
            return 0.0
        left_iris_x = pts[468][0]
        right_iris_x = pts[473][0]
        left_eye_center = (pts[133][0] + pts[362][0]) / 2
        deviation = abs(left_iris_x - left_eye_center) + abs(right_iris_x - left_eye_center)
        eye_width = abs(pts[133][0] - pts[362][0]) + 1e-6
        return float(np.clip(deviation / eye_width, 0, 1))

    def _compute_blink(self, pts: dict) -> float:
        if not all(k in pts for k in [159, 145, 386, 374]):
            return 0.0
        left_ear = abs(pts[159][1] - pts[145][1])
        right_ear = abs(pts[386][1] - pts[374][1])
        return (left_ear + right_ear) / 2

    def _blink_irregularity(self) -> float:
        if len(self._blink_history) < 30:
            return 0.0
        arr = np.array(self._blink_history[-90:])
        std = float(np.std(arr))
        return float(np.clip(std / 5.0, 0, 1))

    def _micro_expression_score(self, aus: Dict[str, float]) -> float:
        deception_aus = ["AU4_brow_lowerer", "AU24_lip_pressor", "AU7_lid_tightener"]
        vals = [aus.get(au, 0) for au in deception_aus]
        return float(np.clip(np.mean(vals) * 1.5, 0, 1)) if vals else 0.0

    def process_video(self, video_path: str) -> FacialResult:
        cap = cv2.VideoCapture(video_path)
        results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            r = self.process_frame(frame)
            if r:
                results.append(r)
        cap.release()

        if not results:
            return FacialResult()

        return FacialResult(
            micro_expression_score=float(np.mean([r.micro_expression_score for r in results])),
            gaze_aversion_score=float(np.mean([r.gaze_aversion_score for r in results])),
            blink_irregularity=float(np.mean([r.blink_irregularity for r in results])),
            lip_compression=float(np.mean([r.lip_compression for r in results])),
            asymmetric_smile=float(np.mean([r.asymmetric_smile for r in results])),
            raw_aus={k: float(np.mean([r.raw_aus.get(k, 0) for r in results]))
                     for k in results[0].raw_aus},
        )
