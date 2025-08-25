# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    YOLO_CONFIG_DIR=/app/ultralytics_config

WORKDIR /app

# System deps for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Ensure runtime dirs exist and are writable
RUN mkdir -p uploads ultralytics_config && chmod -R 775 uploads ultralytics_config

# Render provides PORT env var
EXPOSE 8000

# Start with a single worker to reduce memory; bind to $PORT and allow long cold start
CMD ["sh","-c","gunicorn -w 1 --timeout 180 -b 0.0.0.0:${PORT} app_flask:app"]
