FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install system dependencies for OpenCV and edge device operations
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        htop \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgstreamer1.0-0 \
        libavcodec-extra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /edge

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (single step, more efficient)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code and models
COPY src/ ./src/
# COPY models/ ./models/

# Create non-root user for security
RUN groupadd -r edge && \
    useradd -r -g edge -u 1000 edge && \
    chown -R edge:edge /edge

# Switch to non-root user
USER edge

# Set PATH to use UV's virtual environment
ENV PATH="/edge/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

EXPOSE 8000

CMD ["python", "src/main.py"]