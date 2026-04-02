"""HTTP client for sending detection events to BarnSight API.

Handles event submission with automatic retry via in-memory queue
when the API is unreachable. Runs a background flush thread that
drains the queue when connectivity is restored.
"""

import time
import base64
import threading
from typing import Dict, Optional

import requests

from src.core.logger import logger
from src.core.queue import DetectionQueue
from src.config import settings


class APIClient:
  """Manages communication with the BarnSight API server."""

  def __init__(self, api_url: str = settings.API_URL):
    self.api_url = api_url
    self.queue = DetectionQueue()
    self._is_running = False
    self._flush_thread = None

  def start(self) -> None:
    """Start the background queue flush thread."""
    self._is_running = True
    self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
    self._flush_thread.start()
    logger.info("[+] APIClient initialized and queue flusher started")

  def stop(self) -> None:
    """Stop the flush thread gracefully."""
    self._is_running = False
    if self._flush_thread and self._flush_thread.is_alive():
      self._flush_thread.join(timeout=2.0)
    logger.info("[*] APIClient stopped")

  @staticmethod
  def _normalize_timestamp(payload: Dict) -> Dict:
    """Ensure timestamp ends with 'Z' for UTC ISO 8601 format."""
    if "timestamp" not in payload:
      return payload
    ts = payload["timestamp"]
    if not ts.endswith("Z"):
      if ts.endswith("+00:00"):
        ts = ts[:-6] + "Z"
      else:
        ts += "Z"
      payload["timestamp"] = ts
    return payload

  def _prepare_payload(
    self,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> Dict:
    """Build the final payload, encoding image as base64 if present."""
    prepared = dict(payload)
    if image_bytes:
      prepared["image_snapshot"] = base64.b64encode(image_bytes).decode("utf-8")
    return prepared

  def _get_headers(self) -> Dict[str, str]:
    """Return HTTP headers for API requests."""
    return {
      "Content-Type": "application/json",
      "X-API-Key": settings.API_KEY,
    }

  def send_event(
    self,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> None:
    """Send a detection event to the API. Queues on failure."""
    try:
      prepared = self._prepare_payload(payload, image_bytes)
      prepared = self._normalize_timestamp(prepared)
      response = requests.post(
        self.api_url,
        json=prepared,
        headers=self._get_headers(),
        timeout=5.0,
      )
      response.raise_for_status()
      logger.info(
        f"[+] Successfully sent event for camera {payload.get('camera_id')}"
      )
    except Exception as e:
      logger.error(f"[-] Failed to send event, queuing. Error: {e}")
      self.queue.enqueue(self.api_url, payload, image_bytes)

  def _flush_loop(self) -> None:
    """Background thread: drain queued detections when API is reachable."""
    while self._is_running:
      item = self.queue.dequeue()
      if item is None:
        time.sleep(1.0)
        continue
      try:
        prepared = self._prepare_payload(item["payload"], item["image_bytes"])
        prepared = self._normalize_timestamp(prepared)
        response = requests.post(
          item["endpoint"],
          json=prepared,
          headers=self._get_headers(),
          timeout=5.0,
        )
        response.raise_for_status()
        logger.info("[+] Flushed queued event")
      except Exception as e:
        # Requeue failed item at the back, then back off
        logger.debug(f"[-] Cannot flush event, requeuing: {e}")
        self.queue.requeue(item)
        time.sleep(5.0)
