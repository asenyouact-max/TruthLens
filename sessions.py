from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from db_models import User, Session
from schemas import BaselineUpdate
import json

router = APIRouter()

@router.post("/baseline")
async def update_baseline(payload: BaselineUpdate, db: AsyncSession = Depends(get_db)):
    """Update behavioral baseline for a user (device_id)."""
    result = await db.execute(select(User).where(User.device_id == payload.device_id))
    user = result.scalars().first()
    
    if not user:
        user = User(device_id=payload.device_id)
        db.add(user)
    
    user.baseline_data = payload.baseline_data
    await db.commit()
    return {"status": "success", "message": "Baseline updated"}

@router.get("/history/{device_id}")
async def get_history(device_id: str, db: AsyncSession = Depends(get_db)):
    """Get past analysis sessions for a device."""
    result = await db.execute(
        select(Session)
        .join(User)
        .where(User.device_id == device_id)
        .order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()
    return sessions

@router.get("/")
async def list_all_sessions(db: AsyncSession = Depends(get_db)):
    """List all sessions (admin/dev use)."""
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    sessions = result.scalars().all()
    return sessions
