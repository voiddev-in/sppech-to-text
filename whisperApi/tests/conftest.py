import io
import math
import os
import shutil
import struct
import subprocess
import wave
import pytest
from starlette.testclient import TestClient

# Ensure test settings before importing app
os.environ["WHISPER_API_KEYS"] = "test-secret-key-1,test-secret-key-2"
os.environ["WHISPER_MODELS"] = "tiny"
os.environ["WHISPER_DEFAULT_MODEL"] = "tiny"
os.environ["WHISPER_DEVICE"] = "cpu"
os.environ["WHISPER_COMPUTE_TYPE"] = "int8"
os.environ["WHISPER_MAX_CONCURRENCY"] = "2"
os.environ["WHISPER_MAX_QUEUE"] = "2"
os.environ["WHISPER_MAX_UPLOAD_MB"] = "25"
os.environ["WHISPER_MODELS_DIR"] = "./models"

from app.main import app
from app.engine import engine


def generate_wav_bytes(duration_sec: float = 2.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """Generates a standard PCM 16-bit 16kHz mono WAV file in-memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        total_samples = int(duration_sec * sample_rate)
        frames = bytearray()
        for i in range(total_samples):
            # Synthesize tone with slight envelope to avoid click
            envelope = min(1.0, i / (sample_rate * 0.05), (total_samples - i) / (sample_rate * 0.05))
            val = int(32767.0 * 0.3 * envelope * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wav_file.writeframes(frames)
    return buf.getvalue()


def convert_wav_to_format(wav_bytes: bytes, output_format: str, codec: str) -> bytes:
    """Converts WAV bytes into another format using ffmpeg if present."""
    if not shutil.which("ffmpeg"):
        return b""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        "-c:a",
        codec,
        "-f",
        output_format,
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return proc.stdout
    except Exception:
        return b""


@pytest.fixture(scope="session")
def client():
    # Make sure models are loaded for the test session
    try:
        engine.load_models()
    except Exception:
        pass
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_headers():
    return {"Authorization": "Bearer test-secret-key-1"}


@pytest.fixture(scope="session")
def wav_audio():
    return generate_wav_bytes(duration_sec=2.0)


@pytest.fixture(scope="session")
def webm_audio(wav_audio):
    converted = convert_wav_to_format(wav_audio, "webm", "libopus")
    return converted if converted else wav_audio


@pytest.fixture(scope="session")
def m4a_audio(wav_audio):
    converted = convert_wav_to_format(wav_audio, "ipod", "aac")
    return converted if converted else wav_audio
