# BarnSight Edge Agent Guide

## Project Overview

BarnSight Edge is a Python edge-inference worker for local barn camera monitoring. It reads RTSP or USB camera streams, runs a YOLO object detector locally, deduplicates detections by spatial region, and reports compact events to the BarnSight API.

The system is designed for constrained edge hardware, unstable rural networks, and unattended operation.

## Multi-Camera Model

BarnSight Edge uses one worker/container per camera. A barn can run many camera workers on the same physical edge host, but each worker owns one `STREAM_URL`, one `CAMERA_ID`, one region tracker, one queue, and one log stream.

Use a shared `DEVICE_ID` for the edge host or barn gateway, and unique `CAMERA_ID` values for each physical stream. Do not add a single-process multi-camera loop unless there is a strong operational reason; isolated workers make failures, restarts, queueing, and resource limits easier to manage.

## Core Runtime Flow

```text
Camera stream
  -> StreamHandler keeps latest frame in a background thread
  -> InferenceWorker throttles YOLO inference to INFERENCE_FPS
  -> Detector returns manure detections
  -> RegionTracker suppresses duplicate regions
  -> APIClient sends events through a bounded worker pool
  -> DetectionQueue buffers failed events until API connectivity returns
```

## Key Files

- `src/main.py`: Thin executable entry point.
- `src/inference/worker.py`: InferenceWorker orchestration.
- `src/config.py`: Pydantic settings loaded from environment variables.
- `src/inference/detector.py`: Ultralytics YOLO wrapper.
- `src/core/stream_handler.py`: Threaded camera capture with reconnect backoff.
- `src/core/region_tracker.py`: IoU-based deduplication with TTL and max-entry pruning.
- `src/core/queue.py`: Bounded in-memory offline event queue.
- `src/client/api_client.py`: Retrying HTTP client, TLS controls, bounded async event sending.
- `tests/`: Pytest coverage for queueing, API client behavior, and region tracking.

## Development Commands

```bash
uv sync
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=src --cov-report=term-missing
uv run python -m src.main
```

## Engineering Rules

- Keep edge resource usage bounded. Avoid unbounded threads, queues, caches, or image payloads.
- Do not log secrets. API keys and RTSP URLs can contain credentials.
- Prefer local inference and compact event payloads over raw video upload.
- Preserve 2-space indentation and project formatting conventions.
- Add or update tests when changing queue, API, region tracking, or config behavior.
- Use environment variables for deploy-time behavior instead of hardcoded production settings.

## Agent Tasks

- Before finalizing code changes, run `uv run pytest tests/ -v` and `uv run python -m compileall src`.
- Keep Python indentation at 2 spaces in edited files.
- Keep the multi-camera deployment model documented as one worker/container per camera with shared `DEVICE_ID` and unique `CAMERA_ID`.
- If enabling SQLite queues for multi-camera deployments, use a separate `QUEUE_DB_PATH` per camera worker.

## Security Expectations

- Production API URLs should use `https://` with `REQUIRE_HTTPS=True`.
- Leave `API_VERIFY_TLS=True` except for controlled local testing.
- Keep `.env` files out of source control.
- Treat `API_KEY`, RTSP credentials, model artifacts, and camera identifiers as sensitive operational data.

## Edge Optimization Notes

- Reduce `INFERENCE_FPS` and `IMG_SIZE` first on low-power devices.
- Enable `HALF_PRECISION=True` only on CUDA-capable hardware.
- Keep `EVENT_SEND_WORKERS` low on small CPUs.
- Tune `QUEUE_MAX_SIZE` and `MAX_IMAGE_BYTES` to the device's memory and network budget.
- Keep `ENABLE_DISPLAY=False` in production so detector overlays are skipped.
