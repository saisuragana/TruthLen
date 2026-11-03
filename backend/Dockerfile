FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    TRUTHLEN_DATA_DIR=/data \
    TRUTHLEN_FRONTEND_DIR=/app/frontend

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/requirements-models.txt /app/backend/requirements-models.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt -r /app/backend/requirements-models.txt
COPY backend /app/backend
COPY frontend /app/frontend
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
