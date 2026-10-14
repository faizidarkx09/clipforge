# ClipForge — Docker image (Render / any Docker host)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ffmpeg: the backend shells out to ffmpeg/ffprobe for ingest, reframe and polish.
# libgl1: harmless safety net for OpenCV (headless build doesn't need it, but
# some environments still link against libGL for VideoCapture).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# NOTE: faster-whisper downloads its model on first transcription (~75 MB for
# "tiny", ~150 MB for "base", ~500 MB for "small") into ~/.cache. The host
# needs network access at runtime — but the first analysis will be slower
# while the model downloads. To pre-warm, run one analysis right after deploy.
# The app listens on $PORT (Render injects it; defaults to 7860 elsewhere).

EXPOSE 7860

CMD ["python", "app.py"]
