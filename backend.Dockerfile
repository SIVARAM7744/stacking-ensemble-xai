FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_BACKEND=sqlite \
    SQLITE_PATH=/data/disaster_prediction.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
COPY ml_pipeline/requirements.txt ./ml_pipeline/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend ./backend
COPY ml_pipeline ./ml_pipeline

WORKDIR /app/backend/app

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
