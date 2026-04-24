"""Threaded camera stream handler for real-time frame capture.

Maintains a background thread that continuously reads frames
from a video source (RTSP stream or USB webcam), keeping the
latest frame available for the inference loop.
"""

import time
import threading
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from src.core.logger import logger
from src.config import settings


class StreamHandler:
  """Thread-safe camera stream with auto-reconnection support."""

  def __init__(
    self,
    source: Union[str, int] = 0,
    width: int = 640,
    height: int = 640,
    fps: int = 30,
  ):
    self.source = source
    self.width = width
    self.height = height
    self.fps = fps
    self.cap = self._create_capture(source)
    self._frame: Optional[Tuple[bool, np.ndarray]] = None
    self._lock = threading.Lock()
    self._thread: Optional[threading.Thread] = None
    self._is_running = False
    self._last_frame_at: Optional[float] = None
    self._last_frame_signature: Optional[float] = None
    self._frozen_since: Optional[float] = None
    self.restart_count = 0
    self.consecutive_failures = 0

  def _create_capture(self, source: Union[str, int]) -> cv2.VideoCapture:
    """Open and configure a video capture device."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
      logger.error(f"[x] Failed to open camera source: {source}")
      raise RuntimeError(f"Cannot open camera: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    cap.set(cv2.CAP_PROP_FPS, self.fps)
    return cap

  def start(self) -> None:
    """Begin the background frame capture thread."""
    if self._is_running:
      logger.debug("[*] Camera already running.")
      return
    if not self.cap.isOpened():
      logger.error("[x] Cannot start - camera not found.")
      raise RuntimeError("Camera not found.")
    self._is_running = True
    self._thread = threading.Thread(target=self._update_frame, daemon=True)
    self._thread.start()
    logger.info(f"[+] Camera stream started: {self.source}")

  def stop(self) -> None:
    """Stop the capture thread and release the camera."""
    if not self._is_running:
      return
    logger.info("[*] Stopping camera stream...")
    self._is_running = False
    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=2.0)
      if self._thread.is_alive():
        logger.warning("[!] Thread didn't stop gracefully.")
    if self.cap and self.cap.isOpened():
      self.cap.release()
    with self._lock:
      self._frame = None
      self._last_frame_at = None
      self._last_frame_signature = None
      self._frozen_since = None
    logger.info("[+] Camera stopped.")

  def restart(self) -> None:
    """Stop and re-open the camera stream."""
    logger.info("[*] Restarting camera stream...")
    self.restart_count += 1
    self.stop()
    time.sleep(1.0)
    try:
      self.cap = self._create_capture(self.source)
      self.start()
      logger.info("[+] Camera restarted successfully")
    except Exception as e:
      logger.error(f"[x] Failed to restart camera: {e}")
      raise

  def _update_frame(self) -> None:
    """Background loop: continuously read frames from the camera."""
    reconnect_delay = settings.STREAM_RECONNECT_INITIAL_DELAY
    while self._is_running:
      if not self.cap.isOpened():
        logger.error("[x] Camera disconnected, attempting reconnect...")
        time.sleep(reconnect_delay)
        try:
          self.cap = self._create_capture(self.source)
          reconnect_delay = settings.STREAM_RECONNECT_INITIAL_DELAY
          self.consecutive_failures = 0
          logger.info("[+] Camera reconnected")
        except Exception:
          self.consecutive_failures += 1
          reconnect_delay = min(
            reconnect_delay * 2,
            settings.STREAM_RECONNECT_MAX_DELAY,
          )
        continue
      ret, frame = self.cap.read()
      if not ret:
        self.consecutive_failures += 1
        time.sleep(0.01)
        continue
      signature = self._frame_signature(frame)
      now = time.time()
      with self._lock:
        if self._last_frame_signature == signature:
          self._frozen_since = self._frozen_since or now
        else:
          self._frozen_since = None
        self._last_frame_signature = signature
        self._last_frame_at = now
        self._frame = (ret, frame)
        self.consecutive_failures = 0

  @staticmethod
  def _frame_signature(frame: np.ndarray) -> float:
    """Return a cheap frame signature for frozen-stream detection."""
    sample = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
    return float(sample.mean())

  def read(self) -> Tuple[bool, Optional[np.ndarray]]:
    """Return the latest captured frame."""
    with self._lock:
      if self._frame is None:
        return False, None
      ret, frame = self._frame
      return ret, frame.copy()

  @property
  def is_running(self) -> bool:
    """Whether the capture thread is active."""
    return self._is_running

  @property
  def last_frame_at(self) -> Optional[float]:
    """Unix timestamp for the latest successful frame."""
    with self._lock:
      return self._last_frame_at

  @property
  def is_connected(self) -> bool:
    """Whether OpenCV currently reports the stream as open."""
    return bool(self.cap and self.cap.isOpened())

  @property
  def is_stale(self) -> bool:
    """Whether no frame has arrived within the configured stale window."""
    with self._lock:
      if self._last_frame_at is None:
        return True
      return time.time() - self._last_frame_at > settings.CAMERA_STALE_SECONDS

  @property
  def is_frozen(self) -> bool:
    """Whether the frame signature has been unchanged too long."""
    with self._lock:
      if self._frozen_since is None:
        return False
      return time.time() - self._frozen_since > settings.CAMERA_FROZEN_SECONDS

  def health(self) -> dict:
    """Return camera health for heartbeat and diagnostics."""
    return {
      "camera_connected": self.is_connected,
      "last_frame_at": self.last_frame_at,
      "is_stale": self.is_stale,
      "is_frozen": self.is_frozen,
      "restart_count": self.restart_count,
      "consecutive_failures": self.consecutive_failures,
    }

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()
