from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "TruthLens"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/truthlens"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # S3
    S3_BUCKET: str = "truthlens-videos"
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    # Claude API
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # OpenAI (GPT-4o)
    OPENAI_API_KEY: str = ""

    # DeepSeek R1
    DEEPSEEK_API_KEY: str = ""

    # MiniMax M2.7
    MINIMAX_API_KEY: str = ""

    # Multi-brain settings
    USE_MULTI_BRAIN: bool = True
    BRAIN_AGREEMENT_THRESHOLD: float = 0.20

    # OpenAI (GPT-4o)
    OPENAI_API_KEY: str = ""

    # DeepSeek R1
    DEEPSEEK_API_KEY: str = ""

    # Multi-brain mode — set False to use Claude only (cheaper)
    MULTI_BRAIN_ENABLED: bool = True

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8081"]

    # Analysis thresholds
    DEEPFAKE_CONFIDENCE_THRESHOLD: float = 0.65
    DECEPTION_CONFIDENCE_THRESHOLD: float = 0.60

    # Frame processing
    FRAMES_PER_SECOND_SAMPLE: int = 3  # extract 3 frames/sec for analysis
    AUDIO_CHUNK_SECONDS: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
