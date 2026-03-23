import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import json

import analysis, sessions, health
from config import settings
from database import init_db
from redis_client import init_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
        await init_db()
        await init_redis()
        yield


app = FastAPI(
        title="TruthLens API",
        description="AI-powered lie detection + deepfake detection pipeline",
        version="1.0.0",
        lifespan=lifespan,
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])


@app.websocket("/ws/live/{session_id}")
async def websocket_live_analysis(websocket: WebSocket, session_id: str):
        """Real-time analysis via WebSocket — receives video frames + audio chunks."""
        from websocket_manager import ws_manager
        await ws_manager.connect(websocket, session_id)
        try:
                    while True:
                                    data = await websocket.receive_bytes()
                                    await ws_manager.process_chunk(session_id, data, websocket)
        except WebSocketDisconnect:
                    await ws_manager.disconnect(session_id)


if __name__ == "__main__":
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

