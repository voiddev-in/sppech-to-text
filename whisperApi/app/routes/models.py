from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.config import settings
from app.engine import engine

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get("/models")
async def list_models(_: str = Depends(verify_api_key)):
    # Always include the whisper-1 alias, along with any configured or loaded models
    unique_ids = []
    if "whisper-1" not in unique_ids:
        unique_ids.append("whisper-1")

    for model_name in settings.models_list:
        if model_name not in unique_ids:
            unique_ids.append(model_name)

    for model_name in engine.loaded_models:
        if model_name not in unique_ids:
            unique_ids.append(model_name)

    data = [
        {
            "id": mid,
            "object": "model",
            "created": 1677610602,
            "owned_by": "self-hosted",
        }
        for mid in unique_ids
    ]

    return {
        "object": "list",
        "data": data,
    }
