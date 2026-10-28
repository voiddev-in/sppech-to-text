# Whisper Speech-to-Text API Server

A high-performance, self-hosted, OpenAI-compatible speech-to-text server powered by **FastAPI** and **faster-whisper** (CTranslate2).

This server acts as a drop-in replacement for OpenAI's Whisper audio transcription API, designed to serve the Agentic AI evaluator in `api2` without requiring any changes to `api2` or `web2`.

---

## Features

- **OpenAI API Compatibility**: Implements `GET /v1/models`, `POST /v1/audio/transcriptions`, and `POST /v1/audio/translations`. Works natively with the official OpenAI Python SDK.
- **Model Aliases**: Requests for `whisper-1` are automatically mapped to `WHISPER_DEFAULT_MODEL`.
- **Fast & Efficient**: Powered by CTranslate2 with int8 quantization on CPU and float16 on NVIDIA GPUs.
- **Preloaded Weights**: Models are loaded once at startup and kept in memory for zero per-request initialization latency.
- **Bounded Concurrency & Fast Fail**: Limits concurrent inference tasks (`WHISPER_MAX_CONCURRENCY`) with a queue buffer (`WHISPER_MAX_QUEUE`). Returns `503 Service Unavailable` immediately when overloaded so clients can retry cleanly.
- **Voice Activity Detection (VAD)**: Enabled by default to filter silence, accelerate inference, and avoid hallucinations.
- **Content-Based Format Detection**: Accepts `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`, `ogg`, and `flac`, validated via FFmpeg probing.
- **Strict Privacy**: Uploads are processed in temporary directories and wiped immediately in `finally` blocks. No audio or transcript text is persisted or logged. Logs include only request ID, model, duration, processing time, and real-time factor (RTF).
- **Constant-Time Auth**: Compares API keys using `hmac.compare_digest` to prevent timing attacks.

---

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose (or Python 3.11 with FFmpeg installed).
- NVIDIA Container Toolkit (for GPU execution only).

### 2. Configuration

Copy the sample environment file:

```bash
cp .env-sample .env
```

Edit `.env` to configure your API keys and models (e.g. `WHISPER_API_KEYS=sk-whisper-local-key`).

### 3. Model Download (Optional Ahead of Time)

Model weights (MIT license) can be downloaded ahead of time into `./models` or downloaded automatically on the first server start:

```bash
python scripts/download_models.py tiny base
```

Weights stored in `./models` are gitignored and mounted into the container as a volume.

### 4. Running with Docker Compose

#### CPU Mode (int8)

```bash
docker compose up -d
```

#### GPU Mode (NVIDIA CUDA float16)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Check logs and health status:

```bash
docker compose logs -f
curl http://localhost:8000/health
```

---

## Configuration Reference

All settings are configured via environment variables.

| Variable | Type | Default | Description |
|---|---|---|---|
| `WHISPER_API_KEYS` | String | `sk-whisper-local-key` | Comma-separated list of valid Bearer API keys. |
| `WHISPER_MODELS` | String | `tiny,base` | Comma-separated models to preload at startup (e.g. `tiny,base,small,medium,large-v3`). |
| `WHISPER_DEFAULT_MODEL` | String | `tiny` | Model mapped when client requests the alias `whisper-1`. |
| `WHISPER_DEVICE` | String | `cpu` | Execution device: `cpu` or `cuda`. |
| `WHISPER_COMPUTE_TYPE` | String | `int8` (CPU) / `float16` (CUDA) | Precision: `int8`, `float16`, `float32`, or `default`. |
| `WHISPER_MAX_CONCURRENCY` | Integer | `2` | Maximum concurrent inference workers running at once. |
| `WHISPER_MAX_QUEUE` | Integer | `4` | Maximum queued requests before immediately returning 503. |
| `WHISPER_MAX_UPLOAD_MB` | Integer | `25` | Maximum allowed audio upload size in MB (returns 413 if exceeded). |
| `WHISPER_MAX_AUDIO_SEC` | Integer | `None` | Maximum audio duration in seconds (optional). |
| `WHISPER_BEAM_SIZE` | Integer | `5` | Beam size for decoding (e.g. 5 for accuracy, 1 for greedy/fastest). |
| `WHISPER_VAD` | Boolean | `true` | Suppress silence hallucinations and improve speed via Silero VAD. |
| `WHISPER_MODELS_DIR` | String | `./models` | Directory path where model weights are stored. |
| `PORT` | Integer | `8000` | Port for the HTTP server to listen on. |

---

## Expected Speed & Performance

Real-Time Factor (RTF) measures audio processing time divided by audio duration. Lower is faster: an RTF of 0.10 means 10 seconds of audio is transcribed in 1 second (10x real-time).

| Model | Parameters | Speedup (CPU - 8 cores, int8) | RTF (CPU) | Speedup (GPU - RTX 3090 / T4, float16) | RTF (GPU) |
|---|---|---|---|---|---|
| `tiny` | 39 M | ~15x - 25x | ~0.04 - 0.07 | ~40x - 60x | ~0.015 - 0.025 |
| `base` | 74 M | ~10x - 16x | ~0.06 - 0.10 | ~30x - 45x | ~0.022 - 0.033 |
| `small` | 244 M | ~4x - 6x | ~0.16 - 0.25 | ~18x - 25x | ~0.040 - 0.055 |
| `medium` | 769 M | ~1.5x - 2.5x | ~0.40 - 0.65 | ~8x - 14x | ~0.070 - 0.125 |
| `large-v3` | 1550 M | ~0.8x - 1.2x | ~0.85 - 1.25 | ~5x - 8x | ~0.125 - 0.200 |

*Note: Enabling `WHISPER_VAD=true` significantly accelerates clips with pauses or silence.*

---

## Connect to mm

To use this server as the Speech to Text provider in the `mm` platform:

1. Navigate to **Creds → Agentic AI → Speech to Text**.
2. Configure provider settings:
   - **Base URL**: `http://<host>:<port>/v1` (e.g. `http://localhost:8000/v1` or private DNS).
   - **API Key**: One of the keys defined in `WHISPER_API_KEYS` (e.g. `sk-whisper-local-key`).
3. Click **Fetch models**:
   - The dropdown will populate with available models (including `whisper-1` and individual models).
4. Select a model (e.g. `whisper-1` or `base`).
5. Click **Test voice**:
   - Upload a short voice clip to verify speech-to-text and subtitle generation.
6. **Subtitles Notice**: In Agentic AI evaluation, `response_format="verbose_json"` combined with `timestamp_granularities=["segment"]` is the exact path used to produce subtitles (SRT/VTT) alongside the transcript in a single request.

---

## API Endpoints & Curl Examples

### 1. Health Check (No Auth)

```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "loaded_models": ["tiny", "base"],
  "device": "cpu",
  "queue_depth": 0
}
```

### 2. List Models (Requires Auth)

```bash
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-whisper-local-key"
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1677610602,
      "owned_by": "self-hosted"
    },
    {
      "id": "tiny",
      "object": "model",
      "created": 1677610602,
      "owned_by": "self-hosted"
    }
  ]
}
```

### 3. Audio Transcriptions (`POST /v1/audio/transcriptions`)

#### Default JSON

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

**Response:**
```json
{
  "text": "Hello world this is a test transcription."
}
```

#### Verbose JSON with Segment Timestamps (Agentic AI evaluator mode)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment"
```

**Response:**
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 3.45,
  "text": "Hello world this is a test transcription.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.45,
      "text": "Hello world this is a test transcription.",
      "tokens": [50364, 2425, 1002, 50590],
      "temperature": 0.0,
      "avg_logprob": -0.21,
      "compression_ratio": 1.05,
      "no_speech_prob": 0.01
    }
  ],
  "usage": {
    "type": "duration",
    "seconds": 4
  }
}
```

#### Verbose JSON with Word Timestamps

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment" \
  -F "timestamp_granularities[]=word"
```

#### Plain Text

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=text"
```

#### SubRip Subtitles (SRT)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=srt"
```

#### WebVTT Subtitles (VTT)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=vtt"
```

### 4. Audio Translations (`POST /v1/audio/translations`)

Translates non-English audio directly into English text:

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -H "Authorization: Bearer sk-whisper-local-key" \
  -F "file=@french_audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

---

## OpenAI Python SDK Example

The server behaves identically to OpenAI's audio API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-whisper-local-key",
)

with open("student_response.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

print("Transcript:", result.text)
print("Duration:", result.duration)
print("Segments count:", len(result.segments))
for segment in result.segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
print("Billed seconds:", result.usage.seconds)
```

---

## Running the Test Suite

Run unit and integration tests inside Docker:

```bash
docker compose run --rm whisper pytest -v
```

Or run tests against a live server instance:

```bash
WHISPER_TEST_URL="http://localhost:8000/v1" pytest tests/ -v
```
