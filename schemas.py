from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class SignalBreakdown(BaseModel):
    facial_au_score: float  # 0-1, higher = more stress indicators
    body_language_score: float
    voice_prosody_score: float
    nlp_score: float
    baseline_deviation: Optional[float] = None


class DeepfakeBreakdown(BaseModel):
    visual_artifacts_score: float
    temporal_consistency_score: float
    biological_signals_score: float
    av_sync_score: float
    metadata_score: float
    frequency_artifacts_score: float
    ensemble_score: float  # final combined model score


class AnalysisResult(BaseModel):
    session_id: str

    # AI detection gate
    is_ai_generated: bool
    ai_confidence: float
    deepfake_breakdown: DeepfakeBreakdown

    # Lie detection (only if not AI-generated)
    deception_score: Optional[float] = None  # 0-1
    deception_confidence: Optional[float] = None
    signal_breakdown: Optional[SignalBreakdown] = None

    # Claude reasoning
    explanation: str
    key_observations: List[str]

    # Metadata
    duration_seconds: float
    frames_analyzed: int
    created_at: datetime


class SessionCreate(BaseModel):
    device_id: str
    mode: str = "recorded"  # "live" | "recorded"


class SessionResponse(BaseModel):
    session_id: str
    upload_url: Optional[str] = None  # presigned S3 URL for recorded mode
    ws_url: Optional[str] = None  # WebSocket URL for live mode


class BaselineUpdate(BaseModel):
    device_id: str
    baseline_data: Dict[str, Any]


class LiveChunkResult(BaseModel):
    timestamp_ms: float
    frame_deepfake_score: Optional[float] = None
    frame_deception_indicators: Optional[Dict[str, float]] = None
    running_ai_confidence: float
    running_deception_score: float
