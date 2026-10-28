import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.config import settings
from app.engine import engine
from app.routes.audio import router as audio_router
from app.routes.health import router as health_router
from app.routes.models import router as models_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whisper.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Whisper Speech-to-Text service...")
    try:
        engine.load_models()
        logger.info("All configured models successfully loaded: %s", engine.loaded_models)
    except Exception as e:
        logger.error("Failed to preload models at startup: %s", str(e))
        raise
    yield
    logger.info("Whisper Speech-to-Text service shutting down.")


app = FastAPI(
    title="OpenAI-Compatible Whisper Speech-to-Text API",
    version="1.0.0",
    docs_url=None,  # Disabled for server-to-server security
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    msg = first_error.get("msg", "Validation error")
    loc = first_error.get("loc", [])
    param = loc[-1] if loc else None
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "param": str(param) if param else None,
                "code": "missing_required_parameter" if "required" in msg.lower() else "invalid_parameter",
            }
        },
    )


# Mount routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(audio_router)
