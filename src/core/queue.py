"""In-memory detection queue using thread-safe deque.

Replaces the previous SQLite-backed queue for better performance
on edge hardware. Detections are buffered in memory when the API
is unreachable and flushed automatically when connectivity returns.

Note: Detections are lost on process restart/crash.
"""

import threading
from collections import deque
from typing import Dict, Optional

from src.core.logger import logger


class DetectionQueue:
  """Thread-safe in-memory FIFO queue for buffering detections."""

  def __init__(self, maxsize: int = 1000):
    self.maxsize = maxsize
    self.dropped_count = 0
    self._queue: deque = deque()
    self._lock = threading.Lock()

  def enqueue(
    self,
    endpoint: str,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> bool:
    """Add a detection to the back of the queue.

    Returns True when queued. If the queue is full, the oldest item is dropped
    first so edge devices keep recent detections instead of growing memory.
    """
    with self._lock:
      if len(self._queue) >= self.maxsize:
        self._queue.popleft()
        self.dropped_count += 1
        logger.warning(
          "Detection queue full; dropped oldest item",
          extra={"dropped_count": self.dropped_count, "maxsize": self.maxsize},
        )
      self._queue.append({
        "endpoint": endpoint,
        "payload": payload,
        "image_bytes": image_bytes,
      })
      return True

  def dequeue(self) -> Optional[Dict]:
    """Remove and return the oldest detection, or None if empty."""
    with self._lock:
      if not self._queue:
        return None
      return self._queue.popleft()

  def requeue(self, item: Dict) -> None:
    """Put a failed item back at the end of the queue for retry."""
    with self._lock:
      if len(self._queue) >= self.maxsize:
        self._queue.popleft()
        self.dropped_count += 1
      self._queue.append(item)

  def size(self) -> int:
    """Return the current number of queued detections."""
    with self._lock:
      return len(self._queue)

  def stats(self) -> Dict[str, int]:
    """Return queue metrics useful for health checks and logs."""
    with self._lock:
      return {
        "size": len(self._queue),
        "maxsize": self.maxsize,
        "dropped_count": self.dropped_count,
      }
