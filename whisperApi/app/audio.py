import json
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile, status
from app.config import settings

ALLOWED_CONTAINER_FORMATS = {
    "mp3",
    "mp4",
    "m4a",
    "mov",
    "wav",
    "webm",
    "matroska",
    "ogg",
    "flac",
    "mpeg",
    "mpegvideo",
    "mpga",
    "aac",
    "3gp",
}


def _probe_format_and_duration_ffprobe(file_path: str) -> Tuple[str, float]:
    """Uses ffprobe to detect the actual container format and duration from audio contents."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration",
        "-of",
        "json",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        format_name = fmt.get("format_name", "")
        duration = float(fmt.get("duration", 0.0))
        return format_name, duration
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Could not decode audio file with ffmpeg. Ensure the file is a valid audio clip.",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "invalid_audio",
                }
            },
        )


def _probe_format_and_duration_fallback(file_path: str) -> Tuple[str, float]:
    """Fallback probe using PyAV or basic header inspection when ffprobe is not in PATH."""
    try:
        import av

        with av.open(file_path) as container:
            fmt_name = container.format.name
            duration = 0.0
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
            return fmt_name, duration
    except Exception:
        # Fallback to header inspection
        with open(file_path, "rb") as f:
            header = f.read(64)
        if len(header) < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Empty or corrupted audio file provided.",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "invalid_audio",
                    }
                },
            )
        if header.startswith(b"RIFF") and b"WAVE" in header[:16]:
            return "wav", 1.0
        elif header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xf3"):
            return "mp3", 1.0
        elif header.startswith(b"OggS"):
            return "ogg", 1.0
        elif header.startswith(b"fLaC"):
            return "flac", 1.0
        elif b"ftyp" in header[:16]:
            return "mp4", 1.0
        elif header.startswith(b"\x1a\x45\xdf\xa3"):
            return "matroska,webm", 1.0
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Could not decode audio file with ffmpeg. Unsupported or corrupted audio format.",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "invalid_audio",
                }
            },
        )


def probe_audio(file_path: str) -> Tuple[str, float]:
    if shutil.which("ffprobe") is not None:
        format_name, duration = _probe_format_and_duration_ffprobe(file_path)
    else:
        format_name, duration = _probe_format_and_duration_fallback(file_path)

    # Check format against allowed container formats
    detected_parts = {part.strip().lower() for part in format_name.split(",")}
    if not detected_parts.intersection(ALLOWED_CONTAINER_FORMATS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported audio format '{format_name}'. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac.",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "unsupported_audio_format",
                }
            },
        )

    if settings.whisper_max_audio_sec and duration > settings.whisper_max_audio_sec:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Audio duration ({duration:.1f}s) exceeds maximum allowed duration of {settings.whisper_max_audio_sec}s.",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "audio_too_long",
                }
            },
        )

    return format_name, duration


@asynccontextmanager
async def process_audio_upload(file: UploadFile):
    """
    Saves the uploaded file to a per-request temporary directory,
    enforces WHISPER_MAX_UPLOAD_MB, validates via ffprobe,
    yields (file_path, duration, format_name), and deletes the temp dir in finally.
    """
    import tempfile

    max_bytes = settings.whisper_max_upload_mb * 1024 * 1024
    temp_dir = tempfile.mkdtemp(prefix="whisper_req_")
    try:
        temp_file_path = os.path.join(temp_dir, "audio_input")
        total_bytes = 0
        with open(temp_file_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunk
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": {
                                "message": f"Maximum content size limit ({settings.whisper_max_upload_mb} MB) exceeded.",
                                "type": "invalid_request_error",
                                "param": "file",
                                "code": "file_too_large",
                            }
                        },
                    )
                out.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Empty file provided.",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "empty_file",
                    }
                },
            )

        format_name, duration = probe_audio(temp_file_path)
        yield temp_file_path, duration, format_name
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
