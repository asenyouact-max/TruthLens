from sqlalchemy import Column, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    device_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    baseline_data = Column(JSON, nullable=True)  # stored behavioral baseline
    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    mode = Column(String, default="recorded")  # "live" | "recorded"
    video_s3_key = Column(String, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Results
    is_ai_generated = Column(Boolean, nullable=True)
    ai_confidence = Column(Float, nullable=True)
    deception_score = Column(Float, nullable=True)
    deception_confidence = Column(Float, nullable=True)

    # Detailed signal breakdown (JSON)
    signal_results = Column(JSON, nullable=True)
    claude_explanation = Column(String, nullable=True)

    user = relationship("User", back_populates="sessions")
    signal_frames = relationship("FrameSignal", back_populates="session")


class FrameSignal(Base):
    __tablename__ = "frame_signals"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    timestamp_ms = Column(Float, nullable=False)

    # Per-frame AI detection
    deepfake_score = Column(Float, nullable=True)

    # Per-frame deception signals
    facial_au_data = Column(JSON, nullable=True)
    pose_data = Column(JSON, nullable=True)
    eye_contact_score = Column(Float, nullable=True)
    micro_expression_flags = Column(JSON, nullable=True)

    session = relationship("Session", back_populates="signal_frames")
