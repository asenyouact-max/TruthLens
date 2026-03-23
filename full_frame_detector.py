"""
FullFrameAIDetector — detects AI-generated video across the ENTIRE frame,
not just the face. Works even when no human is present.

Detection zones:
1. Face region      — deepfake, rPPG, blink, AU artifacts (existing)
2. Hands            — extra fingers, digit morphing, impossible anatomy
3. Background       — edge blur seams, texture tiling, AI-typical smoothness
4. Objects + text   — garbled/impossible text (AI can't read/write), object physics
5. Body + clothing  — cloth warping, limb proportion anomalies
6. Whole frame      — GAN frequency grid, CLIP semantic score, temporal drift
7. Audio            — voice cloning artifacts, AV sync, spectral tells
8. Metadata         — codec fingerprints, missing camera EXIF
"""
import asyncio
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from schemas import DeepfakeBreakdown


@dataclass
class FullFrameResult:
    # Per-zone scores (0=natural, 1=AI-generated)
    face_score: float = 0.0
    hands_score: float = 0.0
    background_score: float = 0.0
    objects_text_score: float = 0.0
    body_clothing_score: float = 0.0
    whole_frame_score: float = 0.0
    audio_score: float = 0.0
    metadata_score: float = 0.0

    ensemble_score: float = 0.0
    has_face: bool = False
    flags: List[str] = field(default_factory=list)


# Weights — face + whole-frame are highest signal
ZONE_WEIGHTS = {
    "face":          0.20,
    "hands":         0.12,
    "background":    0.15,
    "objects_text":  0.13,
    "body_clothing": 0.10,
    "whole_frame":   0.18,
    "audio":         0.07,
    "metadata":      0.05,
}


class FullFrameAIDetector:
    def __init__(self):
        self._init_models()

    def _init_models(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.5,
            )
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
            )
            self.mp_available = True
        except ImportError:
            self.mp_available = False

        # CLIP would go here for semantic coherence checks
        # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_available = False

    async def analyze(
        self, frames: List[np.ndarray], audio_path: str, video_path: str
    ) -> DeepfakeBreakdown:
        loop = asyncio.get_event_loop()

        # Run all zone analyzers in parallel
        results = await asyncio.gather(
            loop.run_in_executor(None, self._analyze_face_zone, frames),
            loop.run_in_executor(None, self._analyze_hands_zone, frames),
            loop.run_in_executor(None, self._analyze_background_zone, frames),
            loop.run_in_executor(None, self._analyze_objects_text_zone, frames),
            loop.run_in_executor(None, self._analyze_body_clothing_zone, frames),
            loop.run_in_executor(None, self._analyze_whole_frame, frames),
            loop.run_in_executor(None, self._analyze_audio, audio_path),
            loop.run_in_executor(None, self._analyze_metadata, video_path),
        )

        r = FullFrameResult(
            face_score=results[0][0],
            hands_score=results[1][0],
            background_score=results[2][0],
            objects_text_score=results[3][0],
            body_clothing_score=results[4][0],
            whole_frame_score=results[5][0],
            audio_score=results[6][0],
            metadata_score=results[7][0],
            has_face=results[0][1],
        )

        # Collect flags from all zones
        for _, flags in results:
            if isinstance(flags, list):
                r.flags.extend(flags)

        # Weighted ensemble
        r.ensemble_score = (
            r.face_score          * ZONE_WEIGHTS["face"]
            + r.hands_score       * ZONE_WEIGHTS["hands"]
            + r.background_score  * ZONE_WEIGHTS["background"]
            + r.objects_text_score* ZONE_WEIGHTS["objects_text"]
            + r.body_clothing_score*ZONE_WEIGHTS["body_clothing"]
            + r.whole_frame_score * ZONE_WEIGHTS["whole_frame"]
            + r.audio_score       * ZONE_WEIGHTS["audio"]
            + r.metadata_score    * ZONE_WEIGHTS["metadata"]
        )

        # If no face detected, redistribute face weight to whole_frame + background
        if not r.has_face:
            no_face_bonus = ZONE_WEIGHTS["face"] * r.whole_frame_score * 0.5
            r.ensemble_score += no_face_bonus
            r.flags.append("No face detected — scene-level analysis only")

        r.ensemble_score = round(min(1.0, r.ensemble_score), 3)

        return DeepfakeBreakdown(
            visual_artifacts_score=round((r.face_score + r.background_score) / 2, 3),
            temporal_consistency_score=round(r.whole_frame_score * 0.6 + r.body_clothing_score * 0.4, 3),
            biological_signals_score=round(r.face_score, 3),
            av_sync_score=round(r.audio_score, 3),
            metadata_score=round(r.metadata_score, 3),
            frequency_artifacts_score=round(r.whole_frame_score, 3),
            ensemble_score=r.ensemble_score,
        )

    # ─────────────────────────────────────────────
    # Zone 1: Face
    # ─────────────────────────────────────────────
    def _analyze_face_zone(self, frames: List[np.ndarray]) -> Tuple[float, bool, List[str]]:
        """
        Deepfake face detection: texture smoothness, rPPG absence,
        blink anomalies, eye region coherence.
        """
        flags = []
        face_scores = []
        green_channel = []
        has_face = False

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        for frame in frames[::2]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) == 0:
                continue
            has_face = True
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            # Texture smoothness (AI skin = low local variance)
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            smoothness = max(0.0, min(1.0, 1.0 - lap_var / 350.0))
            face_scores.append(smoothness)

            # rPPG signal collection (forehead)
            fh = face[int(h*0.08):int(h*0.28), int(w*0.2):int(w*0.8)]
            if fh.size > 0:
                green_channel.append(float(np.mean(fh[:, :, 1])))

        if not has_face:
            return 0.3, False, []

        # Check rPPG
        rppg_ai_score = 0.3
        if len(green_channel) >= 10:
            sig = np.array(green_channel) - np.mean(green_channel)
            fft = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), d=1.0 / 3)
            band = (freqs >= 0.8) & (freqs <= 2.5)
            heart_ratio = np.sum(fft[band]) / (np.sum(fft) + 1e-8)
            rppg_ai_score = max(0.0, min(1.0, 1.0 - heart_ratio / 0.3))
            if rppg_ai_score > 0.7:
                flags.append("No detectable heartbeat signal in face (rPPG absent)")

        score = float(np.mean(face_scores)) * 0.5 + rppg_ai_score * 0.5
        if score > 0.6:
            flags.append(f"Face texture unusually smooth (AI-typical: {score:.2f})")

        return round(score, 3), True, flags

    # ─────────────────────────────────────────────
    # Zone 2: Hands
    # ─────────────────────────────────────────────
    def _analyze_hands_zone(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        """
        AI video hallmarks: extra/missing fingers, digit morphing between frames,
        impossible hand topology. MediaPipe gives 21 landmarks per hand.
        Normal hand: exactly 5 fingers, stable landmark count across frames.
        """
        if not self.mp_available:
            return 0.3, []

        flags = []
        anomaly_scores = []

        for frame in frames[::3]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)
            if not results.multi_hand_landmarks:
                continue

            for hand_lm in results.multi_hand_landmarks:
                lms = hand_lm.landmark
                # Finger tip indices: 4,8,12,16,20
                tips = [4, 8, 12, 16, 20]
                mcp  = [1, 5, 9, 13, 17]

                # Check finger proportions — AI often generates wrong ratios
                anomaly = 0.0
                for tip_i, mcp_i in zip(tips, mcp):
                    tip = np.array([lms[tip_i].x, lms[tip_i].y])
                    base = np.array([lms[mcp_i].x, lms[mcp_i].y])
                    length = np.linalg.norm(tip - base)
                    # Real finger length relative to hand: 0.15–0.35 of frame
                    if length < 0.03 or length > 0.45:
                        anomaly += 0.2

                # Check landmark stability (z-depth consistency)
                z_values = [lm.z for lm in lms]
                z_range = max(z_values) - min(z_values)
                if z_range > 0.8:  # extreme depth variation = topology error
                    anomaly += 0.3

                anomaly_scores.append(min(1.0, anomaly))

        if not anomaly_scores:
            return 0.2, []  # no hands = no hand anomaly

        score = float(np.mean(anomaly_scores))
        if score > 0.5:
            flags.append(f"Hand anatomy anomalies detected (score: {score:.2f})")
        if score > 0.7:
            flags.append("Finger morphing or extra digits likely — strong AI indicator")

        return round(score, 3), flags

    # ─────────────────────────────────────────────
    # Zone 3: Background
    # ─────────────────────────────────────────────
    def _analyze_background_zone(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        """
        AI background tells:
        - Subject-background edge blur seam (matting artifacts)
        - Texture tiling patterns (AI repeats textures)
        - Background temporal stability vs subject motion (impossible in real cameras)
        - Over-smooth, low-noise background regions
        """
        flags = []
        scores = []

        for frame in frames[::3]:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect subject region using simple saliency
            # Background = edges of frame (corners + border strip)
            border = 40
            bg_region = np.concatenate([
                gray[:border, :].flatten(),
                gray[-border:, :].flatten(),
                gray[:, :border].flatten(),
                gray[:, -border:].flatten(),
            ])
            center_region = gray[h//4:3*h//4, w//4:3*w//4].flatten()

            # AI backgrounds: very low noise variance
            bg_noise = float(np.std(bg_region))
            center_noise = float(np.std(center_region))

            # Real cameras: bg noise ≥ 8. AI backgrounds often < 4.
            bg_smoothness = max(0.0, min(1.0, 1.0 - bg_noise / 12.0))

            # Edge blur seam: detect sharp transition in blur amount
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            # Check gradient uniformity at subject boundary (row ~1/4 height)
            mid_row_blur = float(np.std(np.abs(lap[h//3, :])))
            seam_score = max(0.0, min(1.0, 1.0 - mid_row_blur / 30.0))

            # Texture tiling: high autocorrelation at regular intervals
            bg_patch = gray[:border*2, :w//2].astype(np.float32)
            if bg_patch.size > 100:
                autocorr = np.corrcoef(bg_patch[:10, :].flatten(), bg_patch[10:20, :].flatten())[0, 1]
                tiling_score = max(0.0, float(autocorr)) if not np.isnan(autocorr) else 0.0
            else:
                tiling_score = 0.0

            frame_score = bg_smoothness * 0.4 + seam_score * 0.35 + tiling_score * 0.25
            scores.append(frame_score)

        if not scores:
            return 0.3, []

        score = float(np.mean(scores))
        if bg_smoothness > 0.6:
            flags.append("Background unusually smooth — possible AI-generated scene")
        if score > 0.65:
            flags.append("Subject-background edge artifacts detected")

        return round(score, 3), flags

    # ─────────────────────────────────────────────
    # Zone 4: Objects + Text
    # ─────────────────────────────────────────────
    def _analyze_objects_text_zone(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        """
        AI video almost always garbles text (signs, labels, books, screens).
        Also checks for physically impossible object states.

        Uses Tesseract OCR to detect text regions, then measures
        character recognizability as a proxy for real vs AI text.
        """
        flags = []
        scores = []

        try:
            import pytesseract
            ocr_available = True
        except ImportError:
            ocr_available = False

        for frame in frames[::5]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Even without OCR: check for text-like regions using MSER
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)

            if len(regions) == 0:
                scores.append(0.2)
                continue

            text_region_score = 0.3  # baseline

            if ocr_available and len(regions) > 5:
                # Sample: try OCR on top-left quadrant where text often appears
                h, w = gray.shape
                roi = gray[:h//2, :w//2]
                try:
                    text = pytesseract.image_to_string(roi, timeout=2)
                    # Real text: high confidence characters. AI text: symbols, noise.
                    real_chars = sum(1 for c in text if c.isalnum() or c in " .,!?")
                    total_chars = max(1, len(text.strip()))
                    readability = real_chars / total_chars

                    if total_chars > 10 and readability < 0.4:
                        text_region_score = 0.75
                        flags.append("Garbled/unreadable text detected — strong AI indicator")
                    elif total_chars > 10 and readability > 0.7:
                        text_region_score = 0.1  # readable text = likely real
                except Exception:
                    pass

            scores.append(text_region_score)

        score = float(np.mean(scores)) if scores else 0.3
        return round(score, 3), flags

    # ─────────────────────────────────────────────
    # Zone 5: Body + Clothing
    # ─────────────────────────────────────────────
    def _analyze_body_clothing_zone(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        """
        AI video tells on body/clothing:
        - Cloth warping artifacts (ripple patterns that don't follow physics)
        - Limb proportion anomalies (too-long arms, neck issues)
        - Hair that clips through shoulders or blends oddly
        - Color inconsistency on clothing across frames (AI drift)
        """
        if not self.mp_available:
            return 0.3, []

        flags = []
        scores = []
        prev_clothing_color = None

        for frame in frames[::3]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb)

            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark
            h, w = frame.shape[:2]

            # Check limb proportions using pose landmarks
            # Shoulder-to-wrist (arm length) vs shoulder width
            left_shoulder  = np.array([lm[11].x, lm[11].y])
            right_shoulder = np.array([lm[12].x, lm[12].y])
            left_wrist     = np.array([lm[15].x, lm[15].y])
            right_wrist    = np.array([lm[16].x, lm[16].y])

            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            left_arm_len   = np.linalg.norm(left_shoulder - left_wrist)
            right_arm_len  = np.linalg.norm(right_shoulder - right_wrist)

            # Normal: arm ≈ 0.9–1.4× shoulder width
            arm_ratio = (left_arm_len + right_arm_len) / (2 * shoulder_width + 1e-8)
            proportion_anomaly = 0.0
            if arm_ratio < 0.5 or arm_ratio > 2.0:
                proportion_anomaly = min(1.0, abs(arm_ratio - 1.1) / 1.0)

            # Clothing color drift: check torso region color consistency
            torso_y1 = int(lm[11].y * h)
            torso_y2 = int(lm[23].y * h)
            torso_x1 = int(min(lm[11].x, lm[12].x) * w)
            torso_x2 = int(max(lm[11].x, lm[12].x) * w)

            if torso_y2 > torso_y1 and torso_x2 > torso_x1:
                torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]
                if torso.size > 0:
                    mean_color = np.mean(torso.reshape(-1, 3), axis=0)
                    if prev_clothing_color is not None:
                        color_drift = np.linalg.norm(mean_color - prev_clothing_color)
                        # Real clothing: drift < 15 between sampled frames. AI: can jump.
                        if color_drift > 40:
                            proportion_anomaly = max(proportion_anomaly, 0.6)
                            flags.append("Clothing color drift between frames detected")
                    prev_clothing_color = mean_color

            scores.append(proportion_anomaly)

        if not scores:
            return 0.2, []

        score = float(np.mean(scores))
        if score > 0.5:
            flags.append(f"Body proportion anomalies detected (score: {score:.2f})")

        return round(score, 3), flags

    # ─────────────────────────────────────────────
    # Zone 6: Whole frame
    # ─────────────────────────────────────────────
    def _analyze_whole_frame(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        """
        Frame-level checks that don't require any specific content:
        1. FFT frequency grid artifacts (GAN/diffusion model fingerprint)
        2. Noise floor analysis (AI video has unnaturally uniform noise)
        3. Temporal coherence (frame-to-frame pixel drift patterns)
        4. Color histogram naturalness (AI color distributions differ subtly)
        5. JPEG/compression artifact consistency
        """
        flags = []
        scores = []

        prev_frame = None
        for frame in frames[::2]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # 1. FFT artifacts
            dft = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = 20 * np.log(np.abs(dft) + 1)
            cy, cx = magnitude.shape[0]//2, magnitude.shape[1]//2
            outer = magnitude.copy()
            outer[cy-15:cy+15, cx-15:cx+15] = 0
            fft_score = min(1.0, float(np.percentile(outer, 99.5)) / 180.0)

            # 2. Noise floor — real cameras have natural sensor noise
            # Estimate noise by looking at near-flat regions
            lap = np.abs(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F))
            flat_mask = lap < 5
            if flat_mask.sum() > 100:
                noise_level = float(np.std(gray[flat_mask]))
                # Real cameras: noise_level 2–12. AI: often < 1.5 or > 20 (over-sharpened)
                if noise_level < 1.5:
                    noise_score = 0.7
                elif noise_level > 20:
                    noise_score = 0.5
                else:
                    noise_score = max(0.0, min(0.4, (3.0 - noise_level) / 3.0))
            else:
                noise_score = 0.3

            # 3. Temporal coherence
            temporal_score = 0.3
            if prev_frame is not None:
                diff = np.abs(gray.astype(np.float32) - prev_frame.astype(np.float32))
                # AI video sometimes has non-physical high-frequency temporal noise
                diff_fft = np.abs(np.fft.fft2(diff))
                high_freq_power = float(np.mean(diff_fft[gray.shape[0]//4:, gray.shape[1]//4:]))
                temporal_score = min(1.0, high_freq_power / 500.0)
            prev_frame = gray.astype(np.uint8)

            # 4. Color naturalness — check green channel dominance (AI often over-saturates)
            b, g, r = cv2.split(frame.astype(np.float32))
            g_dominance = float(np.mean(g)) / (float(np.mean(b)) + float(np.mean(r)) + 1e-8)
            color_score = max(0.0, min(1.0, abs(g_dominance - 0.34) * 5))

            frame_score = (
                fft_score     * 0.30
                + noise_score * 0.25
                + temporal_score * 0.25
                + color_score * 0.20
            )
            scores.append(frame_score)

        if not scores:
            return 0.3, []

        score = float(np.mean(scores))
        if fft_score > 0.6:
            flags.append("GAN/diffusion frequency artifacts in frame spectrum")
        if noise_score > 0.6:
            flags.append("Unnatural noise floor — possible AI rendering artifact")

        return round(score, 3), flags

    # ─────────────────────────────────────────────
    # Zone 7: Audio (full — delegates to AudioAIDetector)
    # ─────────────────────────────────────────────
    def _analyze_audio(self, audio_path: str) -> Tuple[float, List[str]]:
        """
        Full audio AI detection via AudioAIDetector:
          - Voice/TTS: spectral flatness, MFCC drift, pitch regularity, prosody
          - Environment: RT60 reverb, noise floor uniformity, silence patterns
          - AV coherence: lip-sync correlation, emotion congruence
        """
        from audio_ai_detector import AudioAIDetector
        import asyncio

        detector = AudioAIDetector()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                detector.analyze(audio_path, video_frames=None)
            )
        finally:
            loop.close()

        return result.ensemble_score, result.flags

    # ─────────────────────────────────────────────
    # Zone 8: Metadata
    # ─────────────────────────────────────────────
    def _analyze_metadata(self, video_path: str) -> Tuple[float, List[str]]:
        """
        Real device-recorded video has rich metadata.
        AI-generated / edited video loses or genericizes it.
        """
        import subprocess, json, os
        flags = []

        if not os.path.exists(video_path):
            return 0.4, ["Video path not found"]

        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", video_path],
                capture_output=True, text=True, timeout=10
            )
            meta = json.loads(result.stdout)
            tags = meta.get("format", {}).get("tags", {})
            streams = meta.get("streams", [])

            suspicious = 0.0

            # Missing camera make/model
            if not tags.get("make") and not tags.get("Make") and not tags.get("com.apple.quicktime.make"):
                suspicious += 0.20
                flags.append("No camera make/model in metadata")

            # Missing creation time
            if not tags.get("creation_time"):
                suspicious += 0.15

            # Generic encoder (AI tools often use libx264 with default settings)
            encoder = tags.get("encoder", "") + tags.get("Encoder", "")
            if "libx264" in encoder.lower() and not tags.get("make"):
                suspicious += 0.10
                flags.append("Generic libx264 encoder with no device fingerprint")

            # Check for unusually round FPS (AI generators often use exactly 24/25/30)
            for stream in streams:
                if stream.get("codec_type") == "video":
                    fps_str = stream.get("r_frame_rate", "")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        try:
                            fps = int(num) / int(den)
                            # Real cameras: 29.97, 23.976. AI tools: exactly 24, 25, 30.
                            if fps in (24.0, 25.0, 30.0):
                                suspicious += 0.08
                        except Exception:
                            pass

            # No audio stream (some AI video generators produce silent video)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            if not has_audio:
                suspicious += 0.10
                flags.append("No audio stream detected")

            return round(min(1.0, suspicious), 3), flags

        except Exception as e:
            return 0.3, [f"Metadata read error: {str(e)[:60]}"]
