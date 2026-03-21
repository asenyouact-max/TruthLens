from fastapi import APIRouter
from core.config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": "development" if settings.DEBUG else "production",
        "multi_brain": settings.USE_MULTI_BRAIN
    }
