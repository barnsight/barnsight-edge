"""Detection queue backends for offline event buffering.

The default DetectionQueue remains an in-memory FIFO for compatibility. The
SQLite backend persists failed events across process restarts for production
edge deployments.
"""

import json
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Protocol

from src.core.logger import logger


class QueueBackend(Protocol):
  """Common queue interface used by APIClient."""

  maxsize: int
  dropped_count: int

  def enqueue(
    self,
    endpoint: str,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> bool:
    ...

  def dequeue(self) -> Optional[Dict]:
    ...

  def requeue(self, item: Dict, last_error: Optional[str] = None) -> None:
    ...

  def delete(self, item: Dict) -> None:
    ...

  def size(self) -> int:
    ...

  def stats(self) -> Dict[str, int]:
    ...

  def close(self) -> None:
    ...


def _now() -> float:
  return time.time()


class DetectionQueue:
  """Thread-safe in-memory FIFO queue for buffering detections."""

  def __init__(self, maxsize: int = 1000, max_retry_count: int = 0):
    self.maxsize = maxsize
    self.max_retry_count = max_retry_count
    self.dropped_count = 0
    self._queue: deque = deque()
    self._lock = threading.Lock()

  def enqueue(
    self,
    endpoint: str,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> bool:
    """Add a detection to the back of the queue."""
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
        "created_at": _now(),
        "retry_count": 0,
        "last_error": None,
        "last_retry_at": None,
      })
      return True

  def dequeue(self) -> Optional[Dict]:
    """Remove and return the oldest detection, or None if empty."""
    with self._lock:
      if not self._queue:
        return None
      return self._queue.popleft()

  def requeue(self, item: Dict, last_error: Optional[str] = None) -> None:
    """Put a failed item back at the end of the queue for retry."""
    with self._lock:
      retry_count = int(item.get("retry_count") or 0) + 1
      if self.max_retry_count and retry_count > self.max_retry_count:
        self.dropped_count += 1
        logger.warning("Dropping queued event after max retries")
        return
      if len(self._queue) >= self.maxsize:
        self._queue.popleft()
        self.dropped_count += 1
      item["retry_count"] = retry_count
      item["last_error"] = last_error
      item["last_retry_at"] = _now()
      self._queue.append(item)

  def delete(self, item: Dict) -> None:
    """Compatibility no-op; memory dequeue already removes the item."""
    return None

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

  def close(self) -> None:
    """Release backend resources."""
    return None


class SQLiteDetectionQueue:
  """SQLite-backed FIFO queue that survives process restarts."""

  def __init__(
    self,
    db_path: str = "data/events_queue.sqlite3",
    maxsize: int = 1000,
    max_retry_count: int = 0,
    store_images: bool = False,
  ):
    self.db_path = Path(db_path)
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
    self.maxsize = maxsize
    self.max_retry_count = max_retry_count
    self.store_images = store_images
    self.dropped_count = 0
    self._lock = threading.Lock()
    self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._init_db()

  def _init_db(self) -> None:
    with self._conn:
      self._conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_queue (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          endpoint TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          image_bytes BLOB,
          created_at REAL NOT NULL,
          retry_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          last_retry_at REAL
        )
      """)
      self._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_detection_queue_created "
        "ON detection_queue(created_at, id)"
      )

  def _drop_oldest_until_below_limit(self) -> None:
    while self.size() >= self.maxsize:
      row = self._conn.execute(
        "SELECT id FROM detection_queue ORDER BY created_at ASC, id ASC LIMIT 1"
      ).fetchone()
      if row is None:
        return
      self._conn.execute("DELETE FROM detection_queue WHERE id = ?", (row["id"],))
      self.dropped_count += 1

  def enqueue(
    self,
    endpoint: str,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> bool:
    """Persist a detection at the back of the queue."""
    with self._lock, self._conn:
      self._drop_oldest_until_below_limit()
      self._conn.execute(
        """
        INSERT INTO detection_queue (
          endpoint, payload_json, image_bytes, created_at, retry_count
        ) VALUES (?, ?, ?, ?, 0)
        """,
        (
          endpoint,
          json.dumps(payload),
          image_bytes if self.store_images else None,
          _now(),
        ),
      )
      return True

  def dequeue(self) -> Optional[Dict]:
    """Remove and return the oldest detection, or None if empty."""
    with self._lock, self._conn:
      row = self._conn.execute(
        "SELECT * FROM detection_queue ORDER BY created_at ASC, id ASC LIMIT 1"
      ).fetchone()
      if row is None:
        return None
      self._conn.execute("DELETE FROM detection_queue WHERE id = ?", (row["id"],))
      return {
        "id": row["id"],
        "endpoint": row["endpoint"],
        "payload": json.loads(row["payload_json"]),
        "image_bytes": row["image_bytes"],
        "created_at": row["created_at"],
        "retry_count": row["retry_count"],
        "last_error": row["last_error"],
        "last_retry_at": row["last_retry_at"],
      }

  def requeue(self, item: Dict, last_error: Optional[str] = None) -> None:
    """Persist a failed item again for a future retry."""
    retry_count = int(item.get("retry_count") or 0) + 1
    if self.max_retry_count and retry_count > self.max_retry_count:
      self.dropped_count += 1
      logger.warning("Dropping queued event after max retries")
      return

    with self._lock, self._conn:
      self._drop_oldest_until_below_limit()
      self._conn.execute(
        """
        INSERT INTO detection_queue (
          endpoint, payload_json, image_bytes, created_at,
          retry_count, last_error, last_retry_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
          item["endpoint"],
          json.dumps(item["payload"]),
          item.get("image_bytes") if self.store_images else None,
          item.get("created_at") or _now(),
          retry_count,
          last_error,
          _now(),
        ),
      )

  def delete(self, item: Dict) -> None:
    """Delete a queued item by id if it still exists."""
    item_id = item.get("id")
    if item_id is None:
      return
    with self._lock, self._conn:
      self._conn.execute("DELETE FROM detection_queue WHERE id = ?", (item_id,))

  def size(self) -> int:
    """Return the current number of persisted detections."""
    row = self._conn.execute("SELECT COUNT(*) AS count FROM detection_queue").fetchone()
    return int(row["count"])

  def stats(self) -> Dict[str, int]:
    """Return queue metrics useful for health checks and diagnostics."""
    with self._lock:
      return {
        "size": self.size(),
        "maxsize": self.maxsize,
        "dropped_count": self.dropped_count,
      }

  def close(self) -> None:
    """Close the SQLite connection."""
    with self._lock:
      self._conn.close()


def create_detection_queue(
  backend: str,
  maxsize: int,
  db_path: str,
  max_retry_count: int = 0,
  store_images: bool = False,
) -> QueueBackend:
  """Create the configured queue backend."""
  if backend == "sqlite":
    return SQLiteDetectionQueue(
      db_path=db_path,
      maxsize=maxsize,
      max_retry_count=max_retry_count,
      store_images=store_images,
    )
  return DetectionQueue(maxsize=maxsize, max_retry_count=max_retry_count)
