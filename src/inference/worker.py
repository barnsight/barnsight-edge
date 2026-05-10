"""Inference worker orchestration for BarnSight Edge."""

import sys
import time
from typing import Optional

import cv2

from src.client.api_client import APIClient
from src.config import settings
from src.core.events import (
  build_event_payload,
  encode_jpeg,
  select_target_detections,
)
from src.core.logger import logger
from src.core.region_tracker import RegionTracker
from src.core.stream_handler import StreamHandler
from src.inference.detector import Detector


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
    self.camera = self._setup_camera()
    self.detector = self._setup_detector()
    self.region_tracker = self._setup_region_tracker()
    self.api_client = self._setup_api_client()

  def _setup_camera(self) -> StreamHandler:
    """Initialize and start the camera stream."""
    try:
      camera = StreamHandler(
        settings.STREAM_URL,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
        fps=settings.STREAM_FPS,
      )
      camera.start()
      logger.info("[+] Camera initialized")
      return camera
    except Exception as exc:
      logger.error(f"[x] Failed to initialize camera: {exc}")
      sys.exit(1)

  def _setup_detector(self) -> Detector:
    """Load the YOLO detector."""
    try:
      detector = Detector(
        model_path=settings.MODEL_PATH,
        confidence=settings.DETECTION_CONFIDENCE,
        half_precision=settings.HALF_PRECISION,
        img_size=settings.IMG_SIZE,
      )
      logger.info(
        f"[+] Detector loaded from {settings.MODEL_PATH} "
        f"(conf={settings.DETECTION_CONFIDENCE}, "
        f"imgsz={settings.IMG_SIZE}, fp16={settings.HALF_PRECISION})"
      )
      return detector
    except Exception as exc:
      logger.error(f"[x] Failed to load detector: {exc}")
      sys.exit(1)

  def _setup_region_tracker(self) -> RegionTracker:
    """Initialize region-based deduplication tracker."""
    tracker = RegionTracker(
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
    return tracker

  def _setup_api_client(self) -> APIClient:
    """Start API client with background flush thread."""
    api_client = APIClient()
    api_client.start()
    return api_client

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
    """Run inference at target FPS."""
    self._is_running = True
    logger.info(f"[*] Inference worker started (Target FPS: {settings.INFERENCE_FPS})")

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
      if (current_time - last_inference_time) < frame_interval:
        self._handle_display_key()
        time.sleep(0.01)
        continue

      last_inference_time = current_time
      self._run_inference_frame(frame, current_time)
      time.sleep(0.01)

  def _run_inference_frame(self, frame, current_time: float) -> None:
    """Run detector and process results for one frame."""
    try:
      annotated, detections = self.detector.predict(
        frame,
        annotate=settings.ENABLE_DISPLAY,
      )
      self._handle_display(annotated)
      self._handle_detections(detections, annotated, current_time)
    except Exception as exc:
      logger.error(f"[x] Detector error: {exc}")

  def _handle_display_key(self) -> None:
    """Process quit key while inference is throttled."""
    if settings.ENABLE_DISPLAY and cv2.waitKey(1) & 0xFF == ord("q"):
      logger.info("Quitting display window")
      self.stop()

  def _handle_display(self, frame) -> None:
    """Show the debug display and process quit key."""
    if not settings.ENABLE_DISPLAY:
      return
    cv2.imshow("BarnSight Edge - Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      logger.info("Quitting display window")
      self.stop()

  def _handle_detections(
    self,
    detections: list[dict],
    frame,
    current_time: float,
  ) -> None:
    """Deduplicate, encode, and submit detection events."""
    if not self._can_process_detection(detections, current_time):
      return

    selected_detections = select_target_detections(
      detections=detections,
      target_name="manure",
      min_confidence=settings.MIN_CONFIDENCE,
    )
    if not selected_detections:
      return

    if len(selected_detections) > settings.MAX_DETECTIONS_PER_FRAME:
      logger.warning(
        f"Limiting frame events from {len(selected_detections)} "
        f"to {settings.MAX_DETECTIONS_PER_FRAME}"
      )
      selected_detections = selected_detections[:settings.MAX_DETECTIONS_PER_FRAME]

    image_bytes = None
    submitted_count = 0
    for detection in selected_detections:
      bbox = detection["bbox"]
      if self.region_tracker and not self.region_tracker.should_send(bbox):
        logger.debug(
          f"Skipping duplicate region detection. "
          f"Confidence: {detection['confidence']:.2f}"
        )
        continue

      if image_bytes is None:
        image_bytes = encode_jpeg(frame, settings.JPEG_QUALITY)
        if not image_bytes:
          logger.warning("Skipping detection events because JPEG encoding failed")
          return

      if self.region_tracker:
        self.region_tracker.mark_sent(bbox)

      payload = build_event_payload(
        detection=detection,
        camera_id=settings.CAMERA_ID,
        device_id=settings.DEVICE_ID,
        barn_id=settings.BARN_ID,
        zone_id=settings.ZONE_ID,
        model_version=settings.MODEL_VERSION,
        model_path=settings.MODEL_PATH,
        inference_fps=settings.INFERENCE_FPS,
        img_size=settings.IMG_SIZE,
        threshold=settings.MIN_CONFIDENCE,
        edge_app_version=settings.EDGE_APP_VERSION,
        snapshot_mode=settings.SNAPSHOT_MODE,
      )
      if self.api_client:
        self.api_client.submit_event(payload, image_bytes)
      submitted_count += 1

    if submitted_count == 0:
      return

    self.last_detection_time = current_time
    logger.info(
      f"Detected {submitted_count} event(s). "
      f"Image size: {len(image_bytes) if image_bytes else 0} bytes"
    )

  def _can_process_detection(
    self,
    detections: list[dict],
    current_time: float,
  ) -> bool:
    """Return whether detections should be processed this frame."""
    if not detections:
      return False
    return current_time - self.last_detection_time > settings.COOLDOWN_SECONDS
