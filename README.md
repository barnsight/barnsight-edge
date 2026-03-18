# BarnSight Edge Device

Real-time on-device detection of animal excrement for smarter farm hygiene monitoring.

The BarnSight Edge Device is a lightweight, background AI system designed to run directly on farm hardware. It processes live camera feeds locally, detects visible animal excrement using computer vision, and reports structured events to the central BarnSight API — all with low latency and extreme resilience to unstable network conditions.

This repository contains everything required to build, configure, and operate the edge detection component of the BarnSight platform.

## 🚀 Purpose & Philosophy

Modern farms generate visual data constantly, but sending raw video to the cloud is expensive, slow, and unreliable in rural environments.

This project follows a simple principle:
    
**Detect locally. Report intelligently. Store reliably.**

The edge device:
- Performs lightweight, on-device AI inference.
- Minimizes bandwidth usage by only sending JSON events and highly compressed image snippets.
- Continues operating and logging events to disk even during total internet outages.
- Flushes queued events automatically when the connection is restored.
- Acts as a reliable, autonomous sensor requiring zero manual intervention.

## ⚙️ Core Features

- **Hardware Optimized:** Features configurable inference frame rates, internal resolution scaling, and half-precision (FP16) support to run efficiently on low-to-mid tier edge hardware.
- **Offline Queuing:** Uses a local SQLite database to safely buffer events if the central API goes offline.
- **Smart Throttling:** Deduplicates repeated detections across consecutive frames using customizable cooldown periods to prevent API spam.
- **Local Debugging:** Includes an optional OpenCV display overlay for setting up cameras and verifying model performance in real-time.

---

## 🛠 Usage

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

## 🏗 High-Level Architecture

```text
Camera (RTSP / USB)
        ↓
Edge Device (This Repo)
  ├─ Stream Ingestion (Isolated thread, handles drops)
  ├─ YOLO Inference Loop (Hardware throttled)
  ├─ Event Deduplication (Cooldown timers)
  └─ API Client / Offline Queue (SQLite fallback)
        ↓
Central BarnSight API (MongoDB)
```

## License

Licensed under the terms specified in the **LICENSE** file.
