"""
WebSocket endpoint for real-time live video analysis.
Client streams base64-encoded frames + audio chunks.
Server streams back per-frame signal scores.
"""
import asyncio
import base64
import json
import numpy as np
import cv2
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from facial import FacialDetector
from deepfake import DeepfakeDetector

log = structlog.get_logger()
router = APIRouter()

# Per-connection detector instances
class LiveSession:
    def __init__(self):
        self.facial = FacialDetector()
        self.deepfake = DeepfakeDetector()
        self.frame_buffer = []
        self.frame_count = 0
        self.deception_scores = []
        self.ai_scores = []


@router.websocket("/live/{session_id}")
async def live_analysis(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = LiveSession()
    log.info("Live session started", session_id=session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "frame":
                frame_b64 = msg.get("frame")
                if not frame_b64:
                    continue

                # Decode frame
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                session.frame_count += 1
                session.frame_buffer.append(frame)
                if len(session.frame_buffer) > 90:
                    session.frame_buffer.pop(0)

                # Facial analysis every frame
                facial_result = session.facial.process_frame(frame)
                deception_score = 0.0
                alert = None

                if facial_result:
                    deception_score = (
                        facial_result.micro_expression_score * 0.3
                        + facial_result.gaze_aversion_score * 0.25
                        + facial_result.lip_compression * 0.25
                        + facial_result.blink_irregularity * 0.2
                    )
                    session.deception_scores.append(deception_score)

                    # Alert on spike
                    if deception_score > 0.7:
                        alert = "High deception signal detected"

                # Deepfake check every 30 frames (less frequent, more expensive)
                ai_score = 0.0
                if session.frame_count % 30 == 0 and len(session.frame_buffer) >= 10:
                    df_result = session.deepfake._check_rppg(session.frame_buffer[-30:])
                    freq_result = session.deepfake._check_temporal_flicker(session.frame_buffer[-10:])
                    ai_score = (df_result * 0.6 + freq_result * 0.4)
                    session.ai_scores.append(ai_score)
                elif session.ai_scores:
                    ai_score = session.ai_scores[-1]

                # Rolling average for smoothing
                smooth_deception = float(np.mean(session.deception_scores[-10:])) if session.deception_scores else 0.0

                await websocket.send_text(json.dumps({
                    "type": "result",
                    "frame": session.frame_count,
                    "deception_score": round(smooth_deception, 3),
                    "ai_score": round(ai_score, 3),
                    "facial": {
                        "micro_expression": round(facial_result.micro_expression_score, 3) if facial_result else 0,
                        "gaze_aversion": round(facial_result.gaze_aversion_score, 3) if facial_result else 0,
                        "lip_compression": round(facial_result.lip_compression, 3) if facial_result else 0,
                    },
                    "alert": alert,
                }))

            elif msg.get("type") == "end":
                # Final summary
                await websocket.send_text(json.dumps({
                    "type": "summary",
                    "total_frames": session.frame_count,
                    "avg_deception": round(float(np.mean(session.deception_scores)), 3) if session.deception_scores else 0,
                    "avg_ai_score": round(float(np.mean(session.ai_scores)), 3) if session.ai_scores else 0,
                    "peak_deception": round(float(max(session.deception_scores)), 3) if session.deception_scores else 0,
                }))
                break

    except WebSocketDisconnect:
        log.info("Live session disconnected", session_id=session_id)
    except Exception as e:
        log.error("Live session error", error=str(e))
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
