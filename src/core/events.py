"""Event helpers for converting detections into API payloads."""

import datetime
import uuid
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


def select_target_detections(
  detections: List[Dict],
  target_name: str,
  min_confidence: float,
) -> List[Dict]:
  """Return all target detections above threshold, highest confidence first."""
  target_name = target_name.lower()
  matches = []
  for detection in detections:
    name = str(detection.get("name", "")).lower()
    confidence = float(detection.get("confidence", 0.0))
    if name == target_name and confidence >= min_confidence:
      matches.append(detection)
  return sorted(matches, key=lambda detection: detection["confidence"], reverse=True)


def build_event_payload(
  detection: Dict,
  camera_id: str,
  device_id: str,
  *,
  barn_id: str = "",
  zone_id: str = "",
  model_version: str = "",
  model_path: str = "",
  inference_fps: float = 0.0,
  img_size: int = 0,
  threshold: float = 0.0,
  edge_app_version: str = "",
  snapshot_mode: str = "none",
) -> Dict:
  """Build the API event payload for a detection."""
  bbox = detection["bbox"]
  return {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "camera_id": camera_id,
    "device_id": device_id,
    "detected_class": str(detection.get("name", "")),
    "confidence": detection["confidence"],
    "bounding_box": {
      "x": bbox[0],
      "y": bbox[1],
      "width": bbox[2] - bbox[0],
      "height": bbox[3] - bbox[1],
    },
    "model_version": model_version,
    "model_path": model_path,
    "inference_fps": inference_fps,
    "edge_queue_size": 0,
    "img_size": img_size,
    "threshold": threshold,
    "event_id": str(uuid.uuid4()),
    "zone_id": zone_id,
    "barn_id": barn_id,
    "snapshot_mode": snapshot_mode,
    "edge_app_version": edge_app_version,
    "queue_latency_seconds": 0.0,
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


def prepare_detection_events(
  detections: List[Dict],
  frame: np.ndarray,
  target_name: str,
  min_confidence: float,
  camera_id: str,
  device_id: str,
  jpeg_quality: int,
) -> Tuple[List[Tuple[Dict, bytes, Dict]], Optional[bytes]]:
  """Produce one API event per matching target detection."""
  selected_detections = select_target_detections(
    detections,
    target_name,
    min_confidence,
  )
  if not selected_detections:
    return [], None

  image_bytes = encode_jpeg(frame, jpeg_quality)
  if not image_bytes:
    return [], None

  events = [
    (build_event_payload(detection, camera_id, device_id), image_bytes, detection)
    for detection in selected_detections
  ]
  return events, image_bytes
