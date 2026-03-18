import time
import json
import base64
import threading
import requests
from typing import Dict, Optional

from src.core.logger import logger
from src.core.queue import DetectionQueue
from src.config import settings

class APIClient:
    def __init__(self, api_url: str = settings.API_URL):
        self.api_url = api_url
        self.queue = DetectionQueue()
        self._is_running = False
        self._flush_thread = None

    def start(self):
        """Start the queue flushing thread."""
        self._is_running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("[+] APIClient initialized and queue flusher started")

    def stop(self):
        """Stop the queue flushing thread."""
        self._is_running = False
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)
        logger.info("[*] APIClient stopped")

    def _prepare_payload(self, payload: Dict, image_bytes: Optional[bytes] = None) -> Dict:
        """Prepare the payload for sending, optionally adding base64 image."""
        prepared = dict(payload)
        if image_bytes:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            prepared["image_snapshot"] = encoded_image
        return prepared

    def _get_headers(self) -> Dict[str, str]:
        """Get standard HTTP headers including authentication."""
        return {
            "Content-Type": "application/json",
            "X-API-Key": settings.API_KEY
        }

    def send_event(self, payload: Dict, image_bytes: Optional[bytes] = None):
        """Attempt to send an event, queue it if sending fails."""
        try:
            prepared_payload = self._prepare_payload(payload, image_bytes)
            
            # Ensure timestamp strictly follows the UTC format ending with Z as per guide
            if "timestamp" in prepared_payload and not prepared_payload["timestamp"].endswith("Z"):
                 # if it ends with +00:00 (from isoformat), replace it, otherwise append Z
                 if prepared_payload["timestamp"].endswith("+00:00"):
                     prepared_payload["timestamp"] = prepared_payload["timestamp"][:-6] + "Z"
                 else:
                     prepared_payload["timestamp"] += "Z"
            
            response = requests.post(
                self.api_url,
                json=prepared_payload,
                headers=self._get_headers(),
                timeout=5.0
            )
            response.raise_for_status()
            logger.info(f"[+] Successfully sent event for camera {payload.get('camera_id')}")
        except Exception as e:
            logger.error(f"[-] Failed to send event, adding to queue. Error: {e}")
            self.queue.enqueue(self.api_url, payload, image_bytes)

    def _flush_loop(self):
        """Continuously try to send queued events."""
        while self._is_running:
            if self.queue.size() > 0:
                item = self.queue.dequeue()
                if item:
                    try:
                        prepared_payload = self._prepare_payload(item["payload"], item["image_bytes"])
                        
                        # Ensure timestamp strict Z format
                        if "timestamp" in prepared_payload and not prepared_payload["timestamp"].endswith("Z"):
                            if prepared_payload["timestamp"].endswith("+00:00"):
                                prepared_payload["timestamp"] = prepared_payload["timestamp"][:-6] + "Z"
                            else:
                                prepared_payload["timestamp"] += "Z"
                                
                        response = requests.post(
                            item["endpoint"],
                            json=prepared_payload,
                            headers=self._get_headers(),
                            timeout=5.0
                        )
                        response.raise_for_status()
                        self.queue.remove(item["id"])
                        logger.info(f"[+] Flushed queued event {item['id']}")
                    except Exception as e:
                        # If it fails again, we just wait before retrying the same item
                        logger.debug(f"[-] Still cannot flush event {item['id']}: {e}")
                        time.sleep(5.0) # Wait longer if network is down
                        continue # Skip the normal sleep and retry
            
            # Small delay between flush attempts to avoid spinning
            time.sleep(1.0)
