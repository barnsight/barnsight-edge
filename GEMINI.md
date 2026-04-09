# BarnSight Edge Device

## Project Overview
The BarnSight Edge Device is a lightweight, background AI system designed to run directly on farm hardware. It processes live camera feeds locally to detect visible animal excrement using a YOLOv8 computer vision model. It reports structured events to the central BarnSight API, minimizing bandwidth usage by sending JSON events and deduplicating detections. The system is designed to be resilient, offline-capable (via an in-memory queue), and highly optimized for edge hardware (supporting FP16 and variable resolution scaling).

### Core Technologies
- **Language**: Python (>=3.10, <3.12)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **AI Model**: YOLOv8 (Ultralytics) for object detection
- **Libraries**: OpenCV, PyTorch, Pydantic (for configuration), Requests

## Key Directories & Files
- `src/`: Main application source code.
  - `src/main.py`: The entry point for the edge agent (`InferenceWorker`).
  - `src/config.py`: Configuration management using `pydantic-settings`.
  - `src/inference/detector.py`: YOLO model wrapper.
  - `src/core/`: Core modules for queuing, region tracking, stream handling, and logging.
  - `src/client/`: API client for communication with the central BarnSight API.
- `models/`: Contains the trained YOLO weights (e.g., `manure.pt`).
- `scripts/`: Helper scripts for building, running, and Docker management.
- `tests/`: Pytest suite for testing deduplication, queueing, and API client logic.
- `pyproject.toml` & `uv.lock`: Dependency definitions and lockfile.
- `Dockerfile` & `docker-compose.yml`: Configuration for containerized, multi-camera setups.
- `.env.example` / `.env.camera.example`: Environment variable templates.

## Building and Running

### Prerequisites
- Python 3.10 or 3.11
- `uv` installed

### Setup
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Create and configure your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your specific STREAM_URL, API_URL, and API_KEY.
   ```

### Running Locally
You can run the application directly via `uv` or use the provided helper script:
```bash
# Using the helper script
./scripts/run.sh

# Or calling uv directly
uv run python -m src.main
```
*Note: For local debugging with a display overlay, set `ENABLE_DISPLAY=True` in your `.env` file.*

### Running via Docker (Multi-Camera)
The project supports running multiple isolated camera streams using Docker Compose:
1. Create a config file for each camera (e.g., `cp .env.camera.example .env.cam1`).
2. Update the `docker-compose.yml` to define services for each camera.
3. Build and start all cameras:
   ```bash
   docker compose up -d --build
   ```

## Development Conventions
- **Dependency Management**: Strict usage of `uv` for managing dependencies and environments.
- **Testing**: Tests are written using `pytest`. Run tests with:
  ```bash
  uv run pytest tests/ -v
  ```
  Run tests with coverage:
  ```bash
  uv run pytest tests/ -v --cov=src --cov-report=term-missing
  ```
- **Formatting and Linting**: The project uses `ruff` for code formatting and linting (configured in `pyproject.toml` with 2-space indentation).
- **Environment Variables**: Application configuration is strictly handled via environment variables parsed by `pydantic-settings` in `src/config.py`.
- **Hardware Optimization**: Inference FPS, image resolution (`IMG_SIZE`), and precision (`HALF_PRECISION`) are configurable to support low-power devices.
