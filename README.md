# Edge Device

Real-time on-device detection of animal excrement for smarter farm hygiene monitoring.

The BarnSight Edge Device is a lightweight, containerized AI system designed to run directly on farm hardware (edge devices). It processes live camera feeds locally, detects visible animal excrement using computer vision, and reports structured results to a backend service — all with low latency and minimal network dependency.

This repository contains everything required to build, run, and operate the edge detection component of the BarnSight platform.


## 🚀 Purpose & Philosophy

Modern farms generate visual data constantly, but sending raw video to the cloud is expensive, slow, and unreliable in rural environments.

This project follows a simple principle:
    
    Detect locally. Report intelligently. Visualize centrally.

The edge device:

- Performs on-device AI inference
- Minimizes bandwidth usage
- Continues operating even with unstable connectivity
- Acts as a reliable, autonomous sensor in the barn


## What This Edge Device Does

- 📷 Captures live video or image frames from connected cameras

- 🧠 Runs AI-based detection of animal excrement (e.g. manure on floors)

- 🟦 Identifies contaminated regions using bounding boxes or masks

- 📊 Produces structured detection metadata (confidence, location, time)

- 🔗 Sends results to a backend API (e.g. FastAPI server)

- 📝 Logs activity for traceability and diagnostics


⚠️ This system does not perform cleaning actions.
It only detects, visualizes, and reports contamination.

## 🧩High-Level Architecture

```bash
Camera
  ↓
Edge Device (this repo)
  - Video stream handling
  - AI inference
  - Local logging
  ↓
Backend Server (FastAPI)
  ↓
Mobile / Web Client
```

## ⚙️ Core Components

### Stream Handler
- Manages camera input (live streams or frames)
- Handles frame capture and preprocessing
- Designed to work under varying lighting and conditions

### Inference Engine

- Loads optimized AI models
- Runs real-time detection on frames
- Outputs structured detection results (location, confidence)

### Client Module

- Communicates with the backend server
- Sends detection events, metadata, and optional images
- Designed to tolerate intermittent connectivity

### Logging

- Centralized logging for:
    - Detections
    - Errors
    - System health

- Useful for audits and debugging

---

## Usage

This project now runs directly on the host using Python and `uv`.

### Prerequisites

- Python version as specified in `.python-version`
- [`uv`](https://github.com/astral-sh/uv) installed

### Install dependencies

From the project root:

```bash
uv sync
```

### Run the edge app (FastAPI + OpenCV + YOLO)

You can either use the helper script:

```bash
./scripts/run.sh
```

or call `uv` directly:

```bash
uv run python -m src.main
```

The API and web UI will be available at:

```bash
http://localhost:8000/
```

Make sure your `.env` file is configured (e.g. `STREAM_URL`, `MODEL_PATH`, etc.).

# License

Licensed under the terms specified in the **LICENSE** file.





