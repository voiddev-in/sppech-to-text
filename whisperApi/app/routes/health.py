from fastapi import APIRouter
from app.config import settings
from app.concurrency import limiter
from app.engine import engine

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "loaded_models": engine.loaded_models,
        "device": settings.whisper_device,
        "queue_depth": limiter.queue_depth,
    }
