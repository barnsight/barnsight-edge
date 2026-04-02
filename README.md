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
- Continues operating and buffering events in memory even during total internet outages.
- Flushes queued events automatically when the connection is restored.
- Acts as a reliable, autonomous sensor requiring zero manual intervention.

## Core Features

- **Hardware Optimized:** Configurable inference frame rates, internal resolution scaling, and half-precision (FP16) support to run efficiently on low-to-mid tier edge hardware.
- **Offline Queuing:** In-memory FIFO queue buffers events when the API is unreachable; flushes automatically on reconnect.
- **Region-Based Deduplication:** Tracks detection bounding boxes using IoU (Intersection over Union) matching. Prevents duplicate image sends for the same manure spot within a configurable cooldown window.
- **Smart Throttling:** Global cooldown plus per-region cooldown prevents API spam from consecutive frames.
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

Each camera runs in its own isolated container. One host manages all cameras via Docker Compose.

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
- **Stagger starts:** Add `deploy: resources: limits: cpus: '0.5'` per service
- **Limit log size:** The compose file caps logs at 10MB per camera with 3 rotations

## High-Level Architecture

```text
Camera (RTSP / USB)
        ↓
Edge Device (This Repo)
  ├─ Stream Ingestion (Threaded, auto-reconnect with backoff)
  ├─ YOLO Inference Loop (Hardware throttled)
  ├─ Region Tracker (IoU-based deduplication, per-region cooldown)
  ├─ Event Deduplication (Global cooldown timers)
  └─ API Client / Offline Queue (In-memory FIFO, auto-flush)
        ↓
Central BarnSight API (MongoDB + Cloudinary)
```

## Project Structure

```
src/
├── main.py                  # Entry point: InferenceWorker
├── config.py                # Pydantic settings from .env
├── client/
│   └── api_client.py        # HTTP client + offline queue flusher
├── core/
│   ├── logger.py            # JSON-structured logging
│   ├── queue.py             # In-memory detection queue (deque)
│   ├── region_tracker.py    # IoU-based region deduplication
│   └── stream_handler.py    # Threaded camera stream reader
└── inference/
    └── detector.py          # YOLO model wrapper

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
| `INFERENCE_FPS` | `5.0` | Target inference frames per second |
| `HALF_PRECISION` | `False` | Use FP16 (GPU only) |
| `IMG_SIZE` | `640` | Internal inference resolution |
| `API_URL` | `http://localhost:8000/api/v1/events` | Central API endpoint |
| `API_KEY` | `""` | Authentication key (must start with `bs_`) |
| `DEVICE_ID` | `edge-device-01` | Unique device identifier |
| `CAMERA_ID` | `camera-01` | Camera identifier |
| `COOLDOWN_SECONDS` | `1.0` | Global cooldown between events |
| `MIN_CONFIDENCE` | `0.5` | Minimum detection confidence threshold |
| `ENABLE_DISPLAY` | `False` | Show OpenCV debug window |
| `JPEG_QUALITY` | `70` | JPEG compression quality (1-100) |
| `IMAGE_COOLDOWN_SECONDS` | `5.0` | Per-region cooldown before resending image |
| `REGION_OVERLAP_THRESHOLD` | `0.5` | IoU threshold for region matching |

## License

Licensed under the terms specified in the **LICENSE** file.
