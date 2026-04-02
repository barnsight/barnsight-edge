"""In-memory detection queue using thread-safe deque.

Replaces the previous SQLite-backed queue for better performance
on edge hardware. Detections are buffered in memory when the API
is unreachable and flushed automatically when connectivity returns.

Note: Detections are lost on process restart/crash.
"""

import threading
from collections import deque
from typing import Dict, Optional


class DetectionQueue:
  """Thread-safe in-memory FIFO queue for buffering detections."""

  def __init__(self, maxsize: int = 1000):
    self._queue: deque = deque(maxlen=maxsize)
    self._lock = threading.Lock()

  def enqueue(
    self,
    endpoint: str,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> None:
    """Add a detection to the back of the queue."""
    with self._lock:
      self._queue.append({
        "endpoint": endpoint,
        "payload": payload,
        "image_bytes": image_bytes,
      })

  def dequeue(self) -> Optional[Dict]:
    """Remove and return the oldest detection, or None if empty."""
    with self._lock:
      if not self._queue:
        return None
      return self._queue.popleft()

  def requeue(self, item: Dict) -> None:
    """Put a failed item back at the end of the queue for retry."""
    with self._lock:
      self._queue.append(item)

  def size(self) -> int:
    """Return the current number of queued detections."""
    with self._lock:
      return len(self._queue)
