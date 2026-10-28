import io
import pytest


def test_transcription_default_json(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert isinstance(data["text"], str)


def test_transcription_text_format(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert isinstance(response.text, str)


def test_transcription_verbose_json_default(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("task") == "transcribe"
    assert "language" in data
    assert "duration" in data
    assert isinstance(data["duration"], (int, float))
    assert "text" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)
    assert "usage" in data
    assert data["usage"].get("type") == "duration"
    assert isinstance(data["usage"].get("seconds"), int)


def test_transcription_verbose_json_with_word_granularity(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        },
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert "words" in data
    assert isinstance(data["words"], list)


def test_transcription_timestamp_granularities_without_brackets(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": "segment",
        },
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data


def test_transcription_granularities_rejected_on_non_verbose_json(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={
            "model": "whisper-1",
            "response_format": "json",
            "timestamp_granularities[]": "segment",
        },
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["param"] == "timestamp_granularities"
    assert "only be set with response_format 'verbose_json'" in data["error"]["message"]


def test_transcription_srt_format(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "srt"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_transcription_vtt_format(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "vtt"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    assert "text/vtt" in response.headers["content-type"]
    assert response.text.startswith("WEBVTT")


def test_unknown_model_returns_400(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "non-existent-model"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["param"] == "model"
    assert data["error"]["code"] == "model_not_found"


def test_invalid_audio_content_returns_400(client, auth_headers):
    # Upload text disguised as audio
    fake_audio = b"This is just plain text, definitely not an audio file."
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1"},
        files={"file": ("test.wav", fake_audio, "audio/wav")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["param"] == "file"


def test_upload_exceeds_size_limit_returns_413(client, auth_headers, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "whisper_max_upload_mb", 1)  # 1MB limit for test
    oversized_data = b"0" * (1024 * 1024 + 1024)  # > 1MB

    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1"},
        files={"file": ("big.wav", oversized_data, "audio/wav")},
    )
    assert response.status_code == 413
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "file_too_large"


def test_audio_format_webm(client, auth_headers, webm_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("clip.webm", webm_audio, "audio/webm")},
    )
    assert response.status_code == 200
    assert "text" in response.json()


def test_audio_format_m4a(client, auth_headers, m4a_audio):
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("clip.m4a", m4a_audio, "audio/mp4")},
    )
    assert response.status_code == 200
    assert "text" in response.json()
