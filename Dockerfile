FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv export --no-dev --no-hashes --format requirements-txt > requirements.txt

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/barnsight/barnsight-edge"
LABEL org.opencontainers.image.description="BarnSight Edge — multi-camera manure detection"

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 libglib2.0-0 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "src.main"]
