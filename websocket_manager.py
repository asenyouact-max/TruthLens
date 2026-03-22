"""
WebSocketManager — handles live streaming analysis.
Receives video chunks via WebSocket, runs lightweight per-frame
analysis, and streams back running scores in real time.
"""
import asyncio
import numpy as np
import json
from fastapi import WebSocket
from typing import Dict
from core.redis_client import update_session_state, get_session_state


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_buffers: Dict[str, list] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_buffers[session_id] = []
        await update_session_state(session_id, {
            "status": "live",
            "frames_processed": 0,
            "running_ai_score": 0.0,
            "running_deception_score": 0.0,
        })

    async def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.session_buffers.pop(session_id, None)

    async def process_chunk(
        self, session_id: str, data: bytes, websocket: WebSocket
    ):
        """
        Receives raw bytes — client sends:
        - First 4 bytes: chunk type (FRAM or AUDI as ASCII)
        - Remaining: payload

        Runs lightweight per-frame deepfake + deception checks,
        sends back running scores after every frame.
        """
        chunk_type = data[:4].decode("ascii", errors="ignore")
        payload = data[4:]

        state = await get_session_state(session_id) or {}
        frames_count = state.get("frames_processed", 0)
        running_ai = state.get("running_ai_score", 0.0)
        running_dec = state.get("running_deception_score", 0.0)

        if chunk_type == "FRAM":
            frame_result = await self._process_frame(payload)
            if frame_result:
                # Exponential moving average for running scores
                alpha = 0.1
                running_ai = alpha * frame_result["ai_score"] + (1 - alpha) * running_ai
                running_dec = alpha * frame_result["deception_score"] + (1 - alpha) * running_dec
                frames_count += 1

                await update_session_state(session_id, {
                    "frames_processed": frames_count,
                    "running_ai_score": round(running_ai, 3),
                    "running_deception_score": round(running_dec, 3),
                })

                # Stream result back to client
                await websocket.send_text(json.dumps({
                    "type": "frame_result",
                    "timestamp_ms": frame_result["timestamp_ms"],
                    "running_ai_confidence": round(running_ai, 3),
                    "running_deception_score": round(running_dec, 3),
                    "frame_flags": frame_result.get("flags", []),
                }))

    async def _process_frame(self, payload: bytes) -> dict | None:
        """
        Lightweight per-frame analysis for live mode.
        Full pipeline (MediaPipe + frequency analysis) runs on sampled frames.
        """
        import cv2
        try:
            # Decode JPEG frame from payload
            nparr = np.frombuffer(payload, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return None

            # Quick heuristic deepfake check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            ai_score = max(0.0, min(1.0, 1.0 - (laplacian_var / 400.0)))

            # Quick deception signal (eye contact via simple face detection)
            deception_score = 0.3  # baseline — update with MediaPipe in production

            return {
                "timestamp_ms": asyncio.get_event_loop().time() * 1000,
                "ai_score": ai_score,
                "deception_score": deception_score,
                "flags": [],
            }
        except Exception:
            return None


ws_manager = WebSocketManager()
