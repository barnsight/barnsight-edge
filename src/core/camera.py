from typing import Union, Optional, Tuple
import threading
import cv2
import sys

from .logger import logger

class Camera:
  def __init__(self, source: Union[str, int] = ..., width: int = 640, height: int = 640):
    self.cap = cv2.VideoCapture(source)
    self._frame: Optional[Tuple[bool, cv2.Mat]] = None
    self._thread: Optional[threading.Thread] = None
    self.lock = threading.Lock()
    self._is_running = False

    # Setting up camera capture
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  def start(self):
    """
    Start the camera. 
    """
    if not self.cap.isOpened():
      logger.error("Failed connect to the camera.")
      sys.exit()
    self._is_running = True
    logger.info("Camera successfully started.")
    _thread = threading.Thread(target=self._update_frame, daemon=True)
    _thread.start()

  def stop(self) -> None:
    """
    Stop the camera and release resources. 
    """
    self._is_running = False
    if self._thread:
      self._thread.join()
    if self.cap.isOpened():
      self.cap.release()
    logger.info("Camera successfully released and closed.")

  def _update_frame(self) -> None:
    """
    Continuously update the current frame from the camera stream.
    """
    while self._is_running:
      if not self.cap.isOpened():
        continue

      ret, frame = self.cap.read()
      if not ret:
        continue
      
      with self.lock:
        self._frame = (ret, frame)
      
  def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
    """
    Read the camera frame from the camera stream.
    """
    with self.lock:
      if self._frame is None:
        return False, None
      return self._frame