from httpx import AsyncClient, Timeout, Limits, HTTPStatusError
from typing import Optional, Dict, List, Union
from uuid import UUID
import datetime
import asyncio

from core.logger import logger

class APIClient:
  def __init__(
    self,
    base_url: str,
    device_id: Union[str, UUID],
    timeout: float = 30.0,
    max_retries: int = 3,
    headers: Optional[Dict] = None
  ):
    self.base_url = base_url
    self.device_id = str(device_id)
    self.timeout = timeout
    self.max_retries = max_retries
    self.headers = headers

    if not headers:
      self.headers = {
        "X-Device-ID": self.device_id
      }

    self.__client = AsyncClient(
      timeout=Timeout(timeout),
      limits=Limits(max_connections=10, max_keepalive_connections=5),
      headers=self.headers
    ) 

    logger.info(f"Initialized API client for device {self.device_id}")

  async def request(
    self,
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    files: Optional[Dict] = None,
    json: Optional[Dict] = None
  ):
    """Make HTTP request."""
    url = f"{self.base_url}/{endpoint.lstrip('/')}"

    for attempt in range(self.max_retries):
      try:
        response = await self.__client.request(
          method=method,
          url=url,
          data=data,
          params=params,
          files=files,
          json=json
        )
        response.raise_for_status()
        return response.json() if response.text else None 
      except HTTPStatusError as e:
        logger.error(f"HTTP error: {e}")
        if attempt == self.max_retries-1:
          raise
        await asyncio.sleep(2 ** attempt)
      
      except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

    return None

  async def send_detection(
    self,
    endpoint: str,
    detections: List[Dict],
    image_bytes: Optional[bytes] = None
  ):
    """Send detection over HTTP request with optional image data."""
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # We can't send a complex list as form-data easily, so we can send as json if no image,
    # or as form-data with json string payload if image is present.
    payload = {
      "timestamp": timestamp,
      "device_id": self.device_id,
      "detections_count": str(len(detections))
    }
    
    import json as json_lib
    payload["detections"] = json_lib.dumps(detections)

    try:
      if image_bytes:
        files = {"image": ("detection.jpg", image_bytes, "image/jpeg")}
        result = await self.request("POST", endpoint, data=payload, files=files)
      else:
        result = await self.request("POST", endpoint, json=payload)
        
      if result is not None:
        logger.info(f"Successfully sent {len(detections)} detection(s)")
        return True
      return False
    except Exception as e:
      logger.error(f"Failed to send detection: {e}")
      return False
    
  async def close(self):
    """Close the HTTP client."""
    await self.__client.aclose()

  async def __aenter__(self):
    return self
  
  async def __aexit__(self):
    await self.close()

  