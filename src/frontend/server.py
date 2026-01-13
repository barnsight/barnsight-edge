from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import os
import numpy as np
import time
import threading
from typing import Optional

from src.config import settings
from src.core.logger import logger
from src.core.stream_handler import StreamHandler
from src.inference.detector import Detector

from fastrtc import Stream as FastRTCStream  # FastRTC real-time video

app = FastAPI()

# Shared state between camera / detector / FastAPI
camera: Optional[StreamHandler] = None
detector: Optional[Detector] = None

_annotated_frame_lock = threading.Lock()
_latest_annotated_frame: Optional[np.ndarray] = None
_inference_thread: Optional[threading.Thread] = None
_inference_running: bool = False

def _placeholder_frame(message: str = "Waiting for stream...") -> np.ndarray:
    """Generate a placeholder frame with a status message."""
    frame = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, message, (60, settings.FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def _init_components():
    """Lazy-load camera and detector so the app doesn't crash at import time."""
    global camera, detector

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

def _inference_loop() -> None:
    """
    Background loop:
    - Pulls frames from OpenCV (StreamHandler)
    - Runs YOLO detector
    - Stores latest annotated frame for FastAPI to stream
    """
    global _latest_annotated_frame, _inference_running

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
            annotated = detector.predict(frame)
        except Exception as exc:  # pragma: no cover
            logger.error(f"[x] Detector error: {exc}")
            annotated = _placeholder_frame("Detector error")

        with _annotated_frame_lock:
            _latest_annotated_frame = annotated

        # Small delay to prevent overloading CPU/GPU
        time.sleep(0.01)

    logger.info("[*] Inference loop stopped")


# --- FastRTC integration ----------------------------------------------------

def _fastrtc_handler(frame: np.ndarray) -> np.ndarray:
    """
    FastRTC handler for video modality.

    Even though FastRTC will pass us a frame from the browser, our pipeline
    is server-side (RTSP -> OpenCV -> YOLO). So we ignore the incoming frame
    and return the latest annotated server-side frame instead.
    """
    with _annotated_frame_lock:
        out = _latest_annotated_frame.copy() if _latest_annotated_frame is not None else None

    if out is None:
        return _placeholder_frame("Waiting for RTSP...")
    return out


fastrtc_stream = FastRTCStream(
    handler=_fastrtc_handler,
    modality="video",
    mode="send-receive",
)

fastrtc_stream.mount(app)

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

def generate_frames():
    """
    Yield annotated frames for MJPEG streaming.
    FastAPI just reads pre-processed frames from the background thread,
    so camera + YOLO keep running independently of HTTP connections.
    """
    _init_components()  # Ensure components exist if called before startup

    while True:
        # Take the most recent annotated frame from the background worker
        with _annotated_frame_lock:
            frame = _latest_annotated_frame.copy() if _latest_annotated_frame is not None else None

        if frame is None:
            frame = _placeholder_frame("Initializing stream...")

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        # Small pause to avoid tight looping when stream is unavailable
        time.sleep(0.02)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = os.path.join(os.path.dirname(__file__), "web_template.html")
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return f.read()
    return "<h1>Web template not found</h1>"

@app.get("/health")
async def health():
    return {"status": "healthy"}
