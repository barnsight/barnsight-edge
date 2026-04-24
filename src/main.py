"""Entry point for the BarnSight Edge inference worker.

Initializes camera stream, YOLO detector, and API client.
Runs the main detection loop at a configurable FPS target.
"""

import time
import signal
import sys
import datetime
from typing import Optional

import cv2

from src.config import settings
from src.core.logger import logger
from src.core.stream_handler import StreamHandler
from src.core.region_tracker import RegionTracker
from src.inference.detector import Detector
from src.client.api_client import APIClient


class InferenceWorker:
  """Orchestrates camera capture, detection, and event reporting."""

  def __init__(self):
    self.camera: Optional[StreamHandler] = None
    self.detector: Optional[Detector] = None
    self.api_client: Optional[APIClient] = None
    self.region_tracker: Optional[RegionTracker] = None
    self._is_running = False
    self.last_detection_time = 0.0

  def setup(self) -> None:
    """Initialize camera, detector, region tracker, and API client."""
    # Initialize camera stream
    try:
      self.camera = StreamHandler(
        settings.STREAM_URL,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
        fps=settings.STREAM_FPS,
      )
      self.camera.start()
      logger.info("[+] Camera initialized")
    except Exception as exc:
      logger.error(f"[x] Failed to initialize camera: {exc}")
      sys.exit(1)

    # Load YOLO detection model
    try:
      self.detector = Detector(
        model_path=settings.MODEL_PATH,
        half_precision=settings.HALF_PRECISION,
        img_size=settings.IMG_SIZE,
      )
      logger.info(
        f"[+] Detector loaded from {settings.MODEL_PATH} "
        f"(imgsz={settings.IMG_SIZE}, fp16={settings.HALF_PRECISION})"
      )
    except Exception as exc:
      logger.error(f"[x] Failed to load detector: {exc}")
      sys.exit(1)

    # Initialize region-based deduplication tracker
    self.region_tracker = RegionTracker(
      overlap_threshold=settings.REGION_OVERLAP_THRESHOLD,
      cooldown_seconds=settings.IMAGE_COOLDOWN_SECONDS,
      ttl_seconds=settings.REGION_TTL_SECONDS,
      max_entries=settings.REGION_MAX_ENTRIES,
    )
    logger.info(
      f"[+] Region tracker initialized "
      f"(overlap={settings.REGION_OVERLAP_THRESHOLD}, "
      f"cooldown={settings.IMAGE_COOLDOWN_SECONDS}s)"
    )

    # Start API client with background flush thread
    self.api_client = APIClient()
    self.api_client.start()

  def stop(self, *args) -> None:
    """Gracefully shut down all components."""
    logger.info("[*] Stopping inference worker...")
    self._is_running = False
    if self.camera:
      self.camera.stop()
    if self.api_client:
      self.api_client.stop()
    if settings.ENABLE_DISPLAY:
      cv2.destroyAllWindows()
    sys.exit(0)

  def run(self) -> None:
    """Main detection loop — runs inference at target FPS."""
    self._is_running = True
    logger.info(f"[*] Inference worker started (Target FPS: {settings.INFERENCE_FPS})")

    # Calculate minimum interval between inferences
    frame_interval = (
      1.0 / settings.INFERENCE_FPS if settings.INFERENCE_FPS > 0 else 0
    )
    last_inference_time = 0.0

    while self._is_running:
      if not self.camera or not self.detector:
        time.sleep(0.1)
        continue

      ret, frame = self.camera.read()
      if not ret or frame is None:
        time.sleep(0.05)
        continue

      current_time = time.time()

      # Throttle inference to target FPS
      if (current_time - last_inference_time) < frame_interval:
        if settings.ENABLE_DISPLAY:
          if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Quitting display window")
            self.stop()
        time.sleep(0.01)
        continue

      last_inference_time = current_time

      try:
        annotated, detections = self.detector.predict(
          frame,
          annotate=settings.ENABLE_DISPLAY,
        )

        # Show detection window if enabled
        if settings.ENABLE_DISPLAY:
          cv2.imshow("BarnSight Edge - Detection", annotated)
          if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Quitting display window")
            self.stop()

        # Process detections with cooldown to avoid duplicates
        if detections and (
          current_time - self.last_detection_time > settings.COOLDOWN_SECONDS
        ):
          # Find highest-confidence manure detection above threshold
          best_detection = None
          for det in detections:
            if det["name"].lower() == "manure" and det["confidence"] >= settings.MIN_CONFIDENCE:
              if not best_detection or det["confidence"] > best_detection["confidence"]:
                best_detection = det

          if best_detection:
            bbox = best_detection["bbox"]

            # Check region deduplication before sending
            if not self.region_tracker.check_and_mark(bbox):
              logger.debug(
                f"Skipping duplicate region detection. "
                f"Confidence: {best_detection['confidence']:.2f}"
              )
              continue

            self.last_detection_time = current_time

            # Encode annotated frame as JPEG
            ok, buffer = cv2.imencode(
              ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            )
            image_bytes = buffer.tobytes() if ok else None

            # Build event payload
            payload = {
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "camera_id": settings.CAMERA_ID,
              "device_id": settings.DEVICE_ID,
              "confidence": best_detection["confidence"],
              "bounding_box": {
                "x": bbox[0],
                "y": bbox[1],
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
              },
            }

            logger.info(
              f"Detected event. Confidence: {payload['confidence']:.2f}. "
              f"Image size: {len(image_bytes) if image_bytes else 0} bytes"
            )

            # Send event through a bounded executor to avoid unbounded threads.
            self.api_client.submit_event(payload, image_bytes)

      except Exception as exc:
        logger.error(f"[x] Detector error: {exc}")

      time.sleep(0.01)


if __name__ == "__main__":
  worker = InferenceWorker()
  signal.signal(signal.SIGINT, worker.stop)
  signal.signal(signal.SIGTERM, worker.stop)
  worker.setup()
  worker.run()
