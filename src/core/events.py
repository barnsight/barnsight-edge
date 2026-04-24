"""Event helpers for converting detections into API payloads."""

import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def select_best_detection(
  detections: List[Dict],
  target_name: str,
  min_confidence: float,
) -> Optional[Dict]:
  """Return the highest-confidence target detection above threshold."""
  best_detection = None
  target_name = target_name.lower()
  for detection in detections:
    name = str(detection.get("name", "")).lower()
    confidence = float(detection.get("confidence", 0.0))
    if name != target_name or confidence < min_confidence:
      continue
    if not best_detection or confidence > best_detection["confidence"]:
      best_detection = detection
  return best_detection


def build_event_payload(
  detection: Dict,
  camera_id: str,
  device_id: str,
) -> Dict:
  """Build the API event payload for a detection."""
  bbox = detection["bbox"]
  return {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "camera_id": camera_id,
    "device_id": device_id,
    "confidence": detection["confidence"],
    "bounding_box": {
      "x": bbox[0],
      "y": bbox[1],
      "width": bbox[2] - bbox[0],
      "height": bbox[3] - bbox[1],
    },
  }


def encode_jpeg(frame: np.ndarray, quality: int) -> Optional[bytes]:
  """Encode a frame as JPEG bytes."""
  ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
  return buffer.tobytes() if ok else None


def prepare_detection_event(
  detections: List[Dict],
  frame: np.ndarray,
  target_name: str,
  min_confidence: float,
  camera_id: str,
  device_id: str,
  jpeg_quality: int,
) -> Tuple[Optional[Dict], Optional[bytes], Optional[Dict]]:
  """Select the best detection and produce payload plus JPEG bytes."""
  detection = select_best_detection(detections, target_name, min_confidence)
  if not detection:
    return None, None, None
  payload = build_event_payload(detection, camera_id, device_id)
  image_bytes = encode_jpeg(frame, jpeg_quality)
  return payload, image_bytes, detection
