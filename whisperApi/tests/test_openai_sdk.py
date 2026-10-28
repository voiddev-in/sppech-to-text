import os
import socket
import tempfile
import threading
import time
import pytest
import uvicorn
from openai import OpenAI
from app.config import settings
from app.main import app


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def openai_server_url():
    env_url = os.environ.get("WHISPER_TEST_URL")
    if env_url:
        yield env_url
        return

    port = get_free_port()
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)

    url = f"http://127.0.0.1:{port}/v1"
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=3)


@pytest.fixture
def openai_client(openai_server_url):
    api_key = os.environ.get("WHISPER_TEST_API_KEY") or (
        settings.api_keys_list[0] if settings.api_keys_list else "sk-whisper-local-key"
    )
    return OpenAI(base_url=openai_server_url, api_key=api_key)


def test_openai_sdk_models_list(openai_client):
    models_page = openai_client.models.list()
    ids = [m.id for m in models_page.data]
    assert "whisper-1" in ids
    assert "tiny" in ids


def test_openai_sdk_transcription_verbose_json(openai_client, wav_audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_audio)
        tmp.flush()
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        # Assert contract exactly as api2 expects
        assert result.text is not None
        assert result.duration is not None
        assert result.segments is not None

        for seg in result.segments:
            assert hasattr(seg, "start")
            assert hasattr(seg, "end")
            assert hasattr(seg, "text")

        assert result.usage is not None
        assert result.usage.seconds is not None
        assert result.usage.type == "duration"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
