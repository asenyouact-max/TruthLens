"""
VideoProcessor — extracts frames and audio from video files.
Uses OpenCV for frames, ffmpeg for audio.
"""
import cv2
import asyncio
import os
import tempfile
from typing import Tuple, List
import numpy as np
from core.config import settings


class VideoProcessor:
    async def save_temp(self, video_bytes: bytes, session_id: str) -> str:
        path = f"/tmp/{session_id}_input.mp4"
        with open(path, "wb") as f:
            f.write(video_bytes)
        return path

    async def extract(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], str, float, float]:
        """
        Returns:
            frames: list of numpy arrays (BGR)
            audio_path: path to extracted .wav file
            duration: seconds
            fps: original frame rate
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_sync, video_path)

    def _extract_sync(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Sample N frames per second
        sample_interval = max(1, int(fps / settings.FRAMES_PER_SECOND_SAMPLE))
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                frames.append(frame)
            frame_idx += 1

        cap.release()

        # Extract audio using ffmpeg
        audio_path = video_path.replace(".mp4", "_audio.wav")
        os.system(
            f"ffmpeg -i {video_path} -vn -acodec pcm_s16le "
            f"-ar 16000 -ac 1 {audio_path} -y -loglevel quiet"
        )

        return frames, audio_path, duration, fps
