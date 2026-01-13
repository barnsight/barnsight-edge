from typing import Union, Optional, Tuple
import threading
import time
import cv2

from src.core.logger import logger

class StreamHandler:
  def __init__(
      self,
      source: Union[str, int] = 0,
      width: int = 640,
      height: int = 640,
      fps: int = 30
  ):
    self.source = source
    self.width = width
    self.height = height
    self.fps = fps

    self.cap = self._create_capture(source)

    self._frame: Optional[Tuple[bool, cv2.Mat]] = None
    self._lock = threading.Lock()
    
    self._thread: Optional[threading.Thread] = None
    self._is_running = False

  def _create_capture(self, source: Union[str, int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
      logger.error(f"[x] Failed to open camera source: {source}")
      raise RuntimeError(f"Cannot open camera: {source}")
    
    # Setting up configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    cap.set(cv2.CAP_PROP_FPS, self.fps)

    return cap
  
  def start(self) -> None:
    """Start the background frame capture thread. """
    if self._is_running:
      logger.debug("[*] Camera already running.")
      return
    
    if not self.cap.isOpened():
      logger.error("[x] Cannot start - camera not found.")
      raise RuntimeError("Camera not found.")
    
    self._is_running = True
    
    self._thread = threading.Thread(
      target=self._update_frame,
      daemon=True
    )
    self._thread.start()

    logger.info(f"[+] Camera stream started: {self.source}")

  def stop(self) -> None:
    """Stop the camera and release resources."""
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
    """Restart the camera."""

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
    """Continuously update the current frame from the camera stream."""
    while self._is_running:
      if not self.cap.isOpened():
        logger.error("[x] Camera disconnected in capture loop")
        time.sleep(0.1)
        continue

      ret, frame = self.cap.read()
      if not ret:
        continue
      
      with self._lock:
        self._frame = (ret, frame)
      
  def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
    """Read the camera frame from the camera stream."""
    with self._lock:
      if self._frame is None:
        return False, None
      return self._frame

  @property
  def is_running(self) -> bool:
    return self._is_running
      
  def __enter__(self):
    """Context manager support."""
    self.start()
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager cleanup."""
    self.stop()