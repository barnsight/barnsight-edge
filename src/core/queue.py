import sqlite3
import json
import threading
from typing import List, Dict, Optional
from pathlib import Path
from src.core.logger import logger

class DetectionQueue:
    """Local SQLite-backed queue for buffering detections during network failures."""
    def __init__(self, db_path: str = "detections_queue.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        endpoint TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        image_bytes BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def enqueue(self, endpoint: str, payload: Dict, image_bytes: Optional[bytes] = None):
        """Add a detection to the queue."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT INTO detections (endpoint, payload, image_bytes) VALUES (?, ?, ?)",
                        (endpoint, json.dumps(payload), image_bytes)
                    )
            logger.debug(f"Queued detection for endpoint {endpoint}")
        except Exception as e:
            logger.error(f"Failed to enqueue detection: {e}")

    def dequeue(self) -> Optional[Dict]:
        """Get the oldest detection from the queue."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT id, endpoint, payload, image_bytes FROM detections ORDER BY id ASC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row:
                        return {
                            "id": row["id"],
                            "endpoint": row["endpoint"],
                            "payload": json.loads(row["payload"]),
                            "image_bytes": row["image_bytes"]
                        }
                    return None
        except Exception as e:
            logger.error(f"Failed to dequeue detection: {e}")
            return None

    def remove(self, item_id: int):
        """Remove a processed detection from the queue."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM detections WHERE id = ?", (item_id,))
            logger.debug(f"Removed detection {item_id} from queue")
        except Exception as e:
            logger.error(f"Failed to remove detection: {e}")

    def size(self) -> int:
        """Get the current queue size."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM detections")
                    return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0
