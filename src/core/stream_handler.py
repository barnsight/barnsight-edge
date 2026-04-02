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
    logger.info("[+] Camera stopped.")

  def restart(self) -> None:
    """Stop and re-open the camera stream."""
    logger.info("[*] Restarting camera stream...")
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
    reconnect_delay = 0.1
    while self._is_running:
      if not self.cap.isOpened():
        logger.error("[x] Camera disconnected, attempting reconnect...")
        time.sleep(reconnect_delay)
        try:
          self.cap = self._create_capture(self.source)
          reconnect_delay = 0.1  # Reset delay on success
          logger.info("[+] Camera reconnected")
        except Exception:
          reconnect_delay = min(reconnect_delay * 2, 5.0)  # Exponential backoff
        continue
      ret, frame = self.cap.read()
      if not ret:
        continue
      with self._lock:
        self._frame = (ret, frame)

  def read(self) -> Tuple[bool, Optional[np.ndarray]]:
    """Return the latest captured frame."""
    with self._lock:
      if self._frame is None:
        return False, None
      return self._frame

  @property
  def is_running(self) -> bool:
    """Whether the capture thread is active."""
    return self._is_running

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()
