from sqlalchemy import Column, String, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", back_populates="user")
    baselines = relationship("Baseline", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    video_url = Column(String, nullable=True)
    mode = Column(String, default="recorded")  # live | recorded
    status = Column(String, default="pending")  # pending | processing | done | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    result = relationship("AnalysisResult", back_populates="session", uselist=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)

    # Top-level verdicts
    is_ai_generated = Column(Boolean, nullable=True)
    ai_confidence = Column(Float, nullable=True)
    deception_score = Column(Float, nullable=True)  # 0.0 - 1.0

    # Per-signal breakdown (stored as JSON)
    facial_signals = Column(JSON, nullable=True)
    body_signals = Column(JSON, nullable=True)
    voice_signals = Column(JSON, nullable=True)
    nlp_signals = Column(JSON, nullable=True)
    deepfake_signals = Column(JSON, nullable=True)

    # Claude's explanation
    reasoning = Column(String, nullable=True)
    key_indicators = Column(JSON, nullable=True)  # list of strings

    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="result")


class Baseline(Base):
    __tablename__ = "baselines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    subject_name = Column(String, nullable=False)

    # Baseline signal averages
    baseline_data = Column(JSON, nullable=False)  # avg blink rate, pitch range, etc.
    sample_count = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="baselines")
