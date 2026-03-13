from fastapi import FastAPI
import cv2
import os
import numpy as np
import time
import threading
import asyncio
from typing import Optional

from src.config import settings
from src.core.logger import logger
from src.core.stream_handler import StreamHandler
from src.inference.detector import Detector
from src.client.client import APIClient

app = FastAPI()

# Shared state between camera / detector / FastAPI
camera: Optional[StreamHandler] = None
detector: Optional[Detector] = None
api_client: Optional[APIClient] = None

_annotated_frame_lock = threading.Lock()
_latest_annotated_frame: Optional[np.ndarray] = None
_inference_thread: Optional[threading.Thread] = None
_inference_running: bool = False
_last_push_time: float = 0.0

def _placeholder_frame(message: str = "Waiting for stream...") -> np.ndarray:
    """Generate a placeholder frame with a status message."""
    frame = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, message, (60, settings.FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def _init_components():
    """Lazy-load camera, detector, and api client so the app doesn't crash at import time."""
    global camera, detector, api_client

    if camera is None:
        try:
            camera = StreamHandler(
                settings.STREAM_URL,
                width=settings.FRAME_WIDTH,
                height=settings.FRAME_HEIGHT,
            )
            camera.start()
            logger.info("[+] Camera initialized")
        except Exception as exc:  # pragma: no cover - protects runtime
            logger.error(f"[x] Failed to initialize camera: {exc}")
            camera = None

    if detector is None:
        try:
            detector = Detector(model_path=settings.MODEL_PATH)
            logger.info(f"[+] Detector loaded from {settings.MODEL_PATH}")
        except Exception as exc:  # pragma: no cover - protects runtime
            logger.error(f"[x] Failed to load detector: {exc}")
            detector = None
            
    if api_client is None:
        try:
            api_client = APIClient(
                base_url=settings.WEB_API_URL,
                device_id=settings.DEVICE_ID
            )
            logger.info("[+] API Client initialized")
        except Exception as exc:
            logger.error(f"[x] Failed to initialize API client: {exc}")
            api_client = None

def _push_detection_task(client: APIClient, endpoint: str, detections: list, frame: np.ndarray):
    """Background task to push detection image."""
    try:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = buffer.tobytes() if ok else None
        
        async def do_send():
            await client.send_detection(endpoint, detections, image_bytes)
            
        asyncio.run(do_send())
    except Exception as e:
        logger.error(f"[x] Failed in push detection task: {e}")

def _inference_loop() -> None:
    """
    Background loop:
    - Pulls frames from OpenCV (StreamHandler)
    - Runs YOLO detector
    - Stores latest annotated frame for FastAPI to stream
    - Pushes detections to web API if any
    """
    global _latest_annotated_frame, _inference_running, _last_push_time

    logger.info("[*] Inference loop started")
    while _inference_running:
        if not camera or not detector:
            # No components ready yet – just wait a bit
            time.sleep(0.05)
            continue

        ret, frame = camera.read()
        if not ret or frame is None:
            # Camera not ready / RTSP down – show waiting frame
            with _annotated_frame_lock:
                _latest_annotated_frame = _placeholder_frame()
            time.sleep(0.05)
            continue

        try:
            annotated, detections = detector.predict(frame)
            
            # Check if we should push detections (filter by 'manure' or similar if needed)
            # and respect the push interval to avoid spamming the backend
            if detections and api_client:
                current_time = time.time()
                if current_time - _last_push_time >= settings.PUSH_INTERVAL:
                    _last_push_time = current_time
                    threading.Thread(
                        target=_push_detection_task,
                        args=(api_client, "/detections", detections, annotated),
                        daemon=True
                    ).start()
                    
        except Exception as exc:  # pragma: no cover
            logger.error(f"[x] Detector error: {exc}")
            annotated = _placeholder_frame("Detector error")

        with _annotated_frame_lock:
            _latest_annotated_frame = annotated

        # Small delay to prevent overloading CPU/GPU
        time.sleep(0.01)

    logger.info("[*] Inference loop stopped")

@app.on_event("startup")
async def startup_event():
    global _inference_thread, _inference_running

    _init_components()

    # Start background YOLO + OpenCV processing thread
    if not _inference_thread or not _inference_thread.is_alive():
        _inference_running = True
        _inference_thread = threading.Thread(
            target=_inference_loop,
            name="inference-thread",
            daemon=True,
        )
        _inference_thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    global _inference_running

    _inference_running = False
    # Give the thread a moment to exit
    if _inference_thread and _inference_thread.is_alive():
        _inference_thread.join(timeout=2.0)

    if camera:
        camera.stop()

@app.get("/health")
async def health():
    return {"status": "healthy"}
