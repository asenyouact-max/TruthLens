import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from analysis import router as analysis_router
from health import router as health_router
from sessions import router as sessions_router
from database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
        await init_db()
        yield

app = FastAPI(
        title="TruthLens API",
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

app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(sessions_router, prefix="/api/v1/sessions", tags=["sessions"])


@app.get("/")
async def root():
        return {"message": "TruthLens API is live!", "docs": "/docs"}


if __name__ == "__main__":
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
