FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Install linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip git zip unzip wget curl htop libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTE=1 \
    UV_LINK_MODE=copy

# Work directory
WORKDIR /edge

COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install PyTorch CPU version (smaller, faster for edge devices without GPU)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY models/ ./models/

# Set PATH to include UV virtual environment
ENV PATH="/edge/.venv/bin:$PATH"

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match && \
  uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "src/main.py"]