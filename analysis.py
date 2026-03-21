from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from models.schemas import AnalysisResult, SessionCreate, SessionResponse
from services.video_processor import VideoProcessor
from services.analysis_orchestrator import AnalysisOrchestrator
import uuid
import boto3
from core.config import settings

router = APIRouter()


@router.post("/session", response_model=SessionResponse)
async def create_session(payload: SessionCreate, db: AsyncSession = Depends(get_db)):
    """Create analysis session. Returns upload URL (recorded) or WS URL (live)."""
    session_id = str(uuid.uuid4())

    if payload.mode == "recorded":
        # Generate presigned S3 upload URL
        s3 = boto3.client("s3", region_name=settings.AWS_REGION)
        s3_key = f"videos/{session_id}/input.mp4"
        upload_url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": settings.S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600,
        )
        return SessionResponse(session_id=session_id, upload_url=upload_url)

    elif payload.mode == "live":
        ws_url = f"ws://localhost:8000/ws/live/{session_id}"
        return SessionResponse(session_id=session_id, ws_url=ws_url)

    raise HTTPException(status_code=400, detail="mode must be 'recorded' or 'live'")


@router.post("/analyze/{session_id}", response_model=AnalysisResult)
async def analyze_video(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger full analysis pipeline on an uploaded video."""
    orchestrator = AnalysisOrchestrator(session_id=session_id, db=db)
    result = await orchestrator.run_full_pipeline()
    return result


@router.post("/upload-analyze", response_model=AnalysisResult)
async def upload_and_analyze(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Direct upload + analyze in one step (for smaller files / dev use)."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    session_id = str(uuid.uuid4())
    video_bytes = await file.read()

    processor = VideoProcessor()
    temp_path = await processor.save_temp(video_bytes, session_id)

    orchestrator = AnalysisOrchestrator(session_id=session_id, db=db, video_path=temp_path)
    result = await orchestrator.run_full_pipeline()
    return result


@router.get("/result/{session_id}", response_model=AnalysisResult)
async def get_result(session_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch cached analysis result for a session."""
    from core.redis_client import get_session_state
    import json

    state = await get_session_state(session_id)
    if not state or "result" not in state:
        raise HTTPException(status_code=404, detail="Result not found or still processing")

    return AnalysisResult(**state["result"])
