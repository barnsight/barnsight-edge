# BarnSight Edge Device

Real-time on-device detection of animal excrement for smarter farm hygiene monitoring.

The BarnSight Edge Device is a lightweight, background AI system designed to run directly on farm hardware. It processes live camera feeds locally, detects visible animal excrement using computer vision, and reports structured events to the central BarnSight API — all with low latency and resilience to unstable network conditions.

This repository contains everything required to build, configure, and operate the edge detection component of the BarnSight platform.

## Purpose & Philosophy

Modern farms generate visual data constantly, but sending raw video to the cloud is expensive, slow, and unreliable in rural environments.

This project follows a simple principle:

**Detect locally. Report intelligently. Store reliably.**

The edge device:
- Performs lightweight, on-device AI inference.
- Minimizes bandwidth usage by only sending JSON events and highly compressed image snippets.
- Deduplicates detections by spatial region to avoid sending the same manure spot twice.
- Continues operating and buffering events during total internet outages.
- Flushes queued events automatically when the connection is restored.
- Acts as a reliable, autonomous sensor requiring zero manual intervention.

## Core Features

- **Hardware Optimized:** Configurable inference frame rates, internal resolution scaling, and half-precision (FP16) support to run efficiently on low-to-mid tier edge hardware.
- **Offline Queuing:** Bounded memory queue by default, with optional SQLite durability for events that must survive restarts; flushes automatically on reconnect.
- **Bounded Edge Resources:** Queue size, event sender workers, image payload size, stream FPS, and region tracker memory are configurable to prevent runaway RAM/thread usage on small devices.
- **Secure API Transport Controls:** HTTPS enforcement, TLS verification, connect/read timeouts, retry backoff, and API key sanity checks are built into the API client.
- **Region-Based Deduplication:** Tracks detection bounding boxes using IoU (Intersection over Union) matching. Prevents duplicate image sends for the same manure spot within a configurable cooldown window.
- **Smart Throttling:** Global cooldown plus per-region cooldown prevents API spam from consecutive frames.
- **Headless Inference Optimization:** Detection overlays are skipped unless display mode is enabled, reducing CPU use in production.
- **Local Debugging:** Optional OpenCV display overlay for setting up cameras and verifying model performance in real-time.
- **Auto-Reconnection:** Camera stream handler automatically reconnects to RTSP sources with exponential backoff on disconnect.

---

## AI Model

### Model Architecture

BarnSight Edge uses a **YOLOv8** (Ultralytics) object detection model, fine-tuned specifically for detecting animal excrement in barn environments.

### Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8 (Nano/Small) |
| Task | Object Detection |
| Classes | `manure` |
| Input Resolution | 640x640 (configurable: 320, 416, 640) |
| Precision | FP32 (default), FP16 (GPU only) |
| Device | Auto (CUDA if available, else CPU) |

### Model File

The trained weights are stored at `models/manure.pt`. This PyTorch checkpoint file contains the full YOLO model architecture and learned parameters.

### Training

The model was trained on a custom dataset of barn interior images with annotated manure bounding boxes. Key training parameters:

- **Augmentation:** Mosaic, random flip, color jitter, blur
- **Optimizer:** SGD with momentum
- **Loss:** CIoU + BCE
- **Validation:** mAP@0.5, mAP@0.5:0.95

### Performance Tuning

For resource-constrained devices, adjust these settings in `.env`:

```env
IMG_SIZE=320          # Lower resolution = faster inference, less accuracy
INFERENCE_FPS=3.0     # Fewer frames per second = lower CPU usage
HALF_PRECISION=True   # FP16 on CUDA-capable devices (Jetson, etc.)
EVENT_SEND_WORKERS=2  # Keep outbound API work bounded on small CPUs
MAX_IMAGE_BYTES=750000
```

### Adding New Classes

To detect additional objects (e.g., water puddles, feed spillage):

1. Annotate new training data with your preferred tool (Roboflow, CVAT, Label Studio)
2. Export in YOLO format
3. Fine-tune the existing model or train from scratch
4. Replace `models/manure.pt` with your new `.pt` file
5. The detector automatically reads class names from the model

---

## Usage

This project runs directly on the host using Python and the `uv` package manager.

### Prerequisites

- Python 3.10 or 3.11 (as specified in `.python-version`)
- [`uv`](https://github.com/astral-sh/uv) installed

### Installation

1. Install the dependencies:
```bash
uv sync
```

2. Create your configuration file:
```bash
cp .env.example .env
```

3. Update `.env` with your specific settings:
   - `STREAM_URL`: Your camera's RTSP feed or `0` for a local webcam.
   - `API_URL`: The full URL to the central BarnSight API (e.g., `https://api.barnsight.ai/api/v1/events`).
   - `API_KEY`: Your generated authentication key (must start with `bs_`).
   - `REQUIRE_HTTPS=True`: Recommended outside local development.
   - *See `.env.example` for all hardware optimization flags.*

### Run the Edge Agent

You can start the edge agent as a background process or interactively.

Using the helper script:
```bash
./scripts/run.sh
```

Or calling `uv` directly:
```bash
uv run python -m src.main
```

### Local Debugging (Display Mode)

If you want to view the live camera feed with the bounding boxes drawn on screen, set `ENABLE_DISPLAY=True` in your `.env` file before starting the script. Press the `q` key on the window to safely stop the process.

### Running Tests

```bash
uv run pytest tests/ -v
```

With coverage:
```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

## Multi-Camera Setup (Docker)

Each camera runs as its own isolated edge worker. One physical edge host can manage many cameras via Docker Compose, but each camera gets a separate process/container, queue, cooldown state, and log stream.

Use the same `DEVICE_ID` for all cameras attached to the same edge box or barn gateway, and use a unique `CAMERA_ID` for each camera:

```text
Barn
  -> Edge device: edge-barn-01
      -> Worker: barn-01-cam-a
      -> Worker: barn-01-cam-b
      -> Worker: barn-01-cam-c
  -> BarnSight API groups events by device_id, camera_id, barn, and zone
```

This isolation is intentional. A frozen RTSP stream, bad camera password, or high CPU load from one camera should not stop the other cameras from reporting contamination events.

### Prerequisites

- Docker and Docker Compose installed on the host
- All cameras accessible via RTSP or USB

### Quick Start

1. Create a config file for each camera:
```bash
cp .env.camera.example .env.cam1
cp .env.camera.example .env.cam2
cp .env.camera.example .env.cam3
```

2. Edit each `.env.camN` file — at minimum change:
   - `STREAM_URL` — the RTSP URL or device index for that camera
   - `DEVICE_ID` — shared ID for the edge host or barn gateway
   - `CAMERA_ID` — unique name (e.g., `barn-01-cam-a`)
   - `API_KEY` — your BarnSight API key

3. Edit `docker-compose.yml` to add or remove camera services as needed:
```yaml
services:
  cam-barn-01:
    build: .
    container_name: barnsight-cam-01
    env_file: .env.cam1
    restart: unless-stopped

  cam-barn-02:
    build: .
    container_name: barnsight-cam-02
    env_file: .env.cam2
    restart: unless-stopped
```

4. Build and start all cameras:
```bash
docker compose up -d --build
```

### Event Identity

Every camera worker sends events with the same device identity model:

```json
{
  "device_id": "edge-barn-01",
  "camera_id": "barn-01-cam-a",
  "confidence": 0.87,
  "bounding_box": {
    "x": 120,
    "y": 210,
    "width": 80,
    "height": 65
  }
}
```

The API should treat `device_id` as the edge host/gateway and `camera_id` as the physical camera stream. Future barn-floor zones should be scoped per camera because each camera has a different view of the floor.

### Managing Cameras

| Command | Description |
|---|---|
| `docker compose up -d` | Start all cameras in background |
| `docker compose up -d --build` | Rebuild and restart all |
| `docker compose logs -f cam-barn-01` | Follow logs for one camera |
| `docker compose restart cam-barn-02` | Restart one camera only |
| `docker compose stop cam-barn-03` | Stop one camera |
| `docker compose down` | Stop and remove all containers |

### Performance Tips for Low-Power Hosts

- **Reduce per-camera FPS:** Set `INFERENCE_FPS=3.0` in each `.env.camN`
- **Lower resolution:** Set `IMG_SIZE=320` for faster inference
- **Use per-camera queues:** Keep a separate queue file per camera if `QUEUE_BACKEND=sqlite`
- **Stagger starts:** Add `deploy: resources: limits: cpus: '0.5'` per service
- **Limit log size:** The compose file caps logs at 10MB per camera with 3 rotations

## High-Level Architecture

```text
Camera (RTSP / USB)
        ↓
Edge Device (This Repo)
  ├─ Camera Worker A (STREAM_URL + CAMERA_ID + queue)
  ├─ Camera Worker B (STREAM_URL + CAMERA_ID + queue)
  └─ Camera Worker C (STREAM_URL + CAMERA_ID + queue)
        ↓
Central BarnSight API (MongoDB + Cloudinary)
```

Each camera worker contains stream ingestion, YOLO inference, region tracking, event throttling, and API delivery.

## Project Structure

```
src/
├── main.py                  # Thin executable entry point
├── config.py                # Pydantic settings from .env
├── client/
│   └── api_client.py        # HTTP client + offline queue flusher
├── core/
│   ├── logger.py            # JSON-structured logging
│   ├── queue.py             # In-memory or SQLite detection queue
│   ├── region_tracker.py    # IoU-based region deduplication
│   └── stream_handler.py    # Threaded camera stream reader
└── inference/
    ├── detector.py          # YOLO model wrapper
    └── worker.py            # InferenceWorker orchestration

tests/
├── test_region_tracker.py   # Region dedup tests (IoU, cooldown, overlap)
├── test_queue.py            # Queue FIFO, requeue, maxsize tests
└── test_api_client.py       # API send, timestamp, flush tests
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `STREAM_URL` | `0` | Camera source (RTSP URL or webcam index) |
| `MODEL_PATH` | `models/manure.pt` | Path to YOLO weights |
| `FRAME_WIDTH` | `640` | Camera frame width |
| `FRAME_HEIGHT` | `640` | Camera frame height |
| `STREAM_FPS` | `30` | Requested camera capture FPS |
| `STREAM_RECONNECT_INITIAL_DELAY` | `0.1` | Initial reconnect delay after camera disconnect |
| `STREAM_RECONNECT_MAX_DELAY` | `5.0` | Maximum reconnect delay |
| `CAMERA_STALE_SECONDS` | `10.0` | Mark stream stale when no frame arrives in this window |
| `CAMERA_FROZEN_SECONDS` | `30.0` | Mark stream frozen when frame signature does not change |
| `INFERENCE_FPS` | `5.0` | Target inference frames per second |
| `DETECTION_CONFIDENCE` | `0.25` | YOLO candidate threshold before app-level filtering |
| `HALF_PRECISION` | `False` | Use FP16 (GPU only) |
| `IMG_SIZE` | `640` | Internal inference resolution |
| `API_URL` | `http://localhost:8000/api/v1/events` | Central API endpoint |
| `API_KEY` | `""` | Authentication key (must start with `bs_`) |
| `REQUIRE_HTTPS` | `False` | Reject non-HTTPS API URLs when enabled |
| `API_VERIFY_TLS` | `True` | Verify TLS certificates for HTTPS API requests |
| `API_CONNECT_TIMEOUT_SECONDS` | `3.0` | HTTP connection timeout |
| `API_TIMEOUT_SECONDS` | `10.0` | HTTP read timeout |
| `API_MAX_RETRIES` | `2` | Retries for transient API errors |
| `API_BACKOFF_SECONDS` | `0.5` | Retry backoff factor |
| `EVENT_SEND_WORKERS` | `2` | Maximum concurrent outbound event senders |
| `QUEUE_BACKEND` | `memory` | Offline queue backend: `memory` or `sqlite` |
| `QUEUE_DB_PATH` | `data/events_queue.sqlite3` | SQLite queue file path when `QUEUE_BACKEND=sqlite` |
| `QUEUE_MAX_RETRY_COUNT` | `0` | Maximum queue retries; `0` means unlimited |
| `QUEUE_STORE_IMAGES` | `False` | Store image bytes in SQLite queue. Keep disabled by default |
| `QUEUE_MAX_SIZE` | `1000` | Maximum offline queue length; oldest events drop first |
| `MAX_IMAGE_BYTES` | `750000` | Skip image snapshots larger than this limit |
| `DEVICE_ID` | `edge-device-01` | Unique device identifier |
| `CAMERA_ID` | `camera-01` | Camera identifier |
| `COOLDOWN_SECONDS` | `1.0` | Global cooldown between events |
| `MIN_CONFIDENCE` | `0.5` | Minimum confidence required before sending an event |
| `MAX_DETECTIONS_PER_FRAME` | `20` | Maximum manure events created from one frame |
| `ENABLE_DISPLAY` | `False` | Show OpenCV debug window |
| `JPEG_QUALITY` | `70` | JPEG compression quality (1-100) |
| `IMAGE_COOLDOWN_SECONDS` | `5.0` | Per-region cooldown before resending image |
| `REGION_OVERLAP_THRESHOLD` | `0.5` | IoU threshold for region matching |
| `REGION_TTL_SECONDS` | `300.0` | Remove stale tracked regions after this many seconds |
| `REGION_MAX_ENTRIES` | `512` | Maximum tracked regions kept in memory |
| `LOG_LEVEL` | `INFO` | JSON logger level |

## Security Notes

- Use `https://` API endpoints and set `REQUIRE_HTTPS=True` for production deployments.
- Keep `API_VERIFY_TLS=True`; disable it only for controlled local testing.
- Treat `.env`, camera RTSP URLs, and API keys as secrets. Do not commit real credentials.
- Bound `QUEUE_MAX_SIZE`, `EVENT_SEND_WORKERS`, and `MAX_IMAGE_BYTES` for the target hardware to avoid resource exhaustion during network outages.

## License

Licensed under the terms specified in the **LICENSE** file.
