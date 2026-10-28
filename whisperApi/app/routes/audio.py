import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from app.audio import process_audio_upload
from app.auth import verify_api_key
from app.concurrency import limiter
from app.engine import engine
from app.formatting import format_verbose_json, to_srt, to_vtt

logger = logging.getLogger("whisper.access")
router = APIRouter(prefix="/v1/audio", tags=["Audio"])

VALID_RESPONSE_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


def _parse_timestamp_granularities(raw_form) -> List[str]:
    """Accepts timestamp_granularities[] or timestamp_granularities from form data."""
    values = raw_form.getlist("timestamp_granularities[]") + raw_form.getlist("timestamp_granularities")
    granularities: List[str] = []
    for val in values:
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        granularities.extend(str(item).strip() for item in parsed)
                        continue
                except Exception:
                    pass
            for part in val.split(","):
                part = part.strip()
                if part:
                    granularities.append(part)
        elif isinstance(val, list):
            granularities.extend(str(item).strip() for item in val)
    return granularities


async def _handle_audio_request(
    request: Request,
    file: Optional[UploadFile],
    model: str,
    response_format: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: Optional[float],
    task: str = "transcribe",
) -> Response:
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    status_code = status.HTTP_200_OK

    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Audio file is required.",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "missing_required_parameter",
                }
            },
        )

    if not model or not model.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Model parameter is required.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "missing_required_parameter",
                }
            },
        )

    resp_format = (response_format or "json").lower().strip()
    if resp_format not in VALID_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Invalid response_format '{response_format}'. Supported formats: json, verbose_json, text, srt, vtt.",
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": "invalid_parameter",
                }
            },
        )

    # Parse timestamp_granularities
    raw_form = await request.form()
    granularities = _parse_timestamp_granularities(raw_form)

    if granularities and resp_format != "verbose_json":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "timestamp_granularities can only be set with response_format 'verbose_json'",
                    "type": "invalid_request_error",
                    "param": "timestamp_granularities",
                    "code": None,
                }
            },
        )

    # Validate temperature if provided
    if temperature is not None:
        try:
            temperature = float(temperature)
            if temperature < 0.0 or temperature > 1.0:
                raise ValueError()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Temperature must be a float between 0.0 and 1.0.",
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "invalid_parameter",
                    }
                },
            )

    include_words = "word" in granularities
    include_segments = "segment" in granularities if granularities else True
    word_timestamps = include_words

    duration = 0.0
    try:
        # Bounded concurrency check: 503 if queue full
        async with limiter.acquire_slot():
            # Save audio to temp file, validate format via ffprobe, check size limit (413)
            async with process_audio_upload(file) as (temp_path, audio_duration, _):
                text, detected_lang, whisper_duration, segments, words = await engine.transcribe(
                    requested_model=model.strip(),
                    audio_path=temp_path,
                    task=task,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    word_timestamps=word_timestamps,
                )
                duration = whisper_duration or audio_duration

        if resp_format == "json":
            response = JSONResponse(content={"text": text})
        elif resp_format == "verbose_json":
            verbose_data = format_verbose_json(
                task=task,
                language=detected_lang,
                duration=duration,
                text=text,
                segments=segments,
                words=words,
                include_segments=include_segments,
                include_words=include_words,
            )
            response = JSONResponse(content=verbose_data)
        elif resp_format == "text":
            response = Response(content=text, media_type="text/plain; charset=utf-8")
        elif resp_format == "srt":
            response = Response(content=to_srt(segments), media_type="text/plain; charset=utf-8")
        elif resp_format == "vtt":
            response = Response(content=to_vtt(segments), media_type="text/vtt; charset=utf-8")
        else:
            response = JSONResponse(content={"text": text})

        return response

    except HTTPException as e:
        status_code = e.status_code
        raise e
    except Exception as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        logger.error("Internal transcription error (request_id=%s): %s", request_id, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "An internal error occurred during transcription.",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )
    finally:
        proc_time = time.time() - start_time
        rtf = proc_time / max(duration, 0.001) if duration > 0 else 0.0
        # Privacy: log only request id, model, duration, timing, and status. No audio filename or transcript text.
        logger.info(
            "request_id=%s model=%s audio_duration=%.2fs proc_time=%.2fs rtf=%.2f status=%d",
            request_id,
            model,
            duration,
            proc_time,
            rtf,
            status_code,
        )


@router.post("/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile,
    model: str = Form(...),
    response_format: Optional[str] = Form("json"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    _: str = Depends(verify_api_key),
):
    return await _handle_audio_request(
        request=request,
        file=file,
        model=model,
        response_format=response_format or "json",
        language=language,
        prompt=prompt,
        temperature=temperature,
        task="transcribe",
    )


@router.post("/translations")
async def create_translation(
    request: Request,
    file: UploadFile,
    model: str = Form(...),
    response_format: Optional[str] = Form("json"),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    _: str = Depends(verify_api_key),
):
    return await _handle_audio_request(
        request=request,
        file=file,
        model=model,
        response_format=response_format or "json",
        language=None,
        prompt=prompt,
        temperature=temperature,
        task="translate",
    )
