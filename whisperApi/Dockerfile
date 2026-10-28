FROM python:3.11-slim

# Install system dependencies: ffmpeg for audio decoding and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Ensure models directory exists
RUN mkdir -p /models

# Copy application source code, scripts, and tests
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WHISPER_MODELS_DIR=/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
