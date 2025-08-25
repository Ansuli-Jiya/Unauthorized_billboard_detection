# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Ensure uploads dir exists and is writable
RUN mkdir -p uploads && chmod -R 775 uploads

# Render expects a web service listening on $PORT; we'll default to 8000
ENV PORT=8000
EXPOSE 8000

# Start with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app_flask:app"]
