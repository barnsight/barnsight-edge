"""HTTP client for sending detection events to BarnSight API.

Handles event submission with automatic retry via in-memory queue
when the API is unreachable. Runs a background flush thread that
drains the queue when connectivity is restored.
"""

import time
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.core.logger import logger
from src.core.queue import create_detection_queue
from src.config import settings


class APIClient:
  """Manages communication with the BarnSight API server."""

  def __init__(self, api_url: str = settings.API_URL):
    self.api_url = api_url
    self.queue = create_detection_queue(
      backend=settings.QUEUE_BACKEND,
      maxsize=settings.QUEUE_MAX_SIZE,
      db_path=settings.QUEUE_DB_PATH,
      max_retry_count=settings.QUEUE_MAX_RETRY_COUNT,
      store_images=settings.QUEUE_STORE_IMAGES,
    )
    self.session = requests.Session()
    self.session.mount("http://", self._build_adapter())
    self.session.mount("https://", self._build_adapter())
    self._timeout = (
      settings.API_CONNECT_TIMEOUT_SECONDS,
      settings.API_TIMEOUT_SECONDS,
    )
    self._executor = ThreadPoolExecutor(
      max_workers=settings.EVENT_SEND_WORKERS,
      thread_name_prefix="api-send",
    )
    self._is_running = False
    self._flush_thread = None
    self._warn_if_insecure()

  @staticmethod
  def _build_adapter() -> HTTPAdapter:
    """Build a retrying requests adapter for transient network failures."""
    retry = Retry(
      total=settings.API_MAX_RETRIES,
      connect=settings.API_MAX_RETRIES,
      read=settings.API_MAX_RETRIES,
      status=settings.API_MAX_RETRIES,
      backoff_factor=settings.API_BACKOFF_SECONDS,
      status_forcelist=(429, 500, 502, 503, 504),
      allowed_methods=frozenset({"POST"}),
      raise_on_status=False,
    )
    return HTTPAdapter(max_retries=retry, pool_maxsize=settings.EVENT_SEND_WORKERS + 2)

  def _warn_if_insecure(self) -> None:
    """Surface insecure deployment configuration without logging secrets."""
    scheme = urlparse(self.api_url).scheme
    if settings.REQUIRE_HTTPS and scheme != "https":
      raise ValueError("API_URL must use https:// when REQUIRE_HTTPS=True")
    if scheme != "https":
      logger.warning("API_URL is not HTTPS; use TLS outside local development")
    if not settings.API_KEY:
      logger.warning("API_KEY is empty; API authentication will likely fail")
    if settings.API_KEY and not settings.API_KEY.startswith("bs_"):
      logger.warning("API_KEY does not use the expected bs_ prefix")

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
    self._executor.shutdown(wait=False, cancel_futures=True)
    self.queue.close()
    self.session.close()
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
      if len(image_bytes) > settings.MAX_IMAGE_BYTES:
        logger.warning(
          "Skipping image snapshot because it exceeds MAX_IMAGE_BYTES",
          extra={"image_bytes": len(image_bytes), "limit": settings.MAX_IMAGE_BYTES},
        )
        return prepared
      b64_str = base64.b64encode(image_bytes).decode("utf-8")
      prepared["image_snapshot"] = f"data:image/jpeg;base64,{b64_str}"
    return prepared

  def _get_headers(self) -> Dict[str, str]:
    """Return HTTP headers for API requests."""
    return {
      "Content-Type": "application/json",
      "X-API-Key": settings.API_KEY,
    }

  def _post_json(self, url: str, payload: Dict) -> requests.Response:
    """Post JSON using shared timeout, TLS, headers, and retry session."""
    return self.session.post(
      url,
      json=payload,
      headers=self._get_headers(),
      timeout=self._timeout,
      verify=settings.API_VERIFY_TLS,
    )

  def send_event(
    self,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> None:
    """Send a detection event to the API. Queues on failure."""
    try:
      prepared = self._prepare_payload(payload, image_bytes)
      prepared = self._normalize_timestamp(prepared)
      response = self._post_json(self.api_url, prepared)
      response.raise_for_status()
      logger.info(
        f"[+] Successfully sent event for camera {payload.get('camera_id')}"
      )
    except Exception as e:
      logger.error(f"[-] Failed to send event, queuing. Error: {e}")
      self.queue.enqueue(self.api_url, payload, image_bytes)

  def send_heartbeat(self, url: str, payload: Dict) -> bool:
    """Send heartbeat without queuing on failure."""
    try:
      response = self._post_json(url, payload)
      response.raise_for_status()
      return True
    except Exception as exc:
      logger.debug(f"Heartbeat send failed: {exc}")
      return False

  def get_remote_config(self, url: str) -> Optional[Dict]:
    """Fetch remote device config without mutating local secrets."""
    try:
      response = self.session.get(
        url,
        headers=self._get_headers(),
        timeout=self._timeout,
        verify=settings.API_VERIFY_TLS,
      )
      response.raise_for_status()
      return response.json()
    except Exception as exc:
      logger.debug(f"Remote config fetch failed: {exc}")
      return None

  def submit_event(
    self,
    payload: Dict,
    image_bytes: Optional[bytes] = None,
  ) -> None:
    """Submit an event for bounded background sending."""
    self._executor.submit(self.send_event, payload, image_bytes)

  def _flush_loop(self) -> None:
    """Background thread: drain queued detections when API is reachable."""
    while self._is_running:
      item = self.queue.dequeue()
      if item is None:
        time.sleep(1.0)
        continue
      try:
        payload = dict(item["payload"])
        created_at = item.get("created_at")
        if created_at:
          payload["queue_latency_seconds"] = max(0.0, time.time() - created_at)
        prepared = self._prepare_payload(payload, item["image_bytes"])
        prepared = self._normalize_timestamp(prepared)
        response = self._post_json(item["endpoint"], prepared)
        response.raise_for_status()
        self.queue.delete(item)
        logger.info("[+] Flushed queued event")
      except Exception as e:
        # Requeue failed item at the back, then back off
        logger.debug(f"[-] Cannot flush event, requeuing: {e}")
        self.queue.requeue(item, last_error=str(e))
        time.sleep(5.0)
