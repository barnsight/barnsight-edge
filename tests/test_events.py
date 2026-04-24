"""Tests for detection event helper functions."""

import numpy as np

from src.core.events import (
  build_event_payload,
  encode_jpeg,
  prepare_detection_event,
  select_best_detection,
)


class TestSelectBestDetection:
  """Tests for target detection selection."""

  def test_selects_highest_confidence_target(self):
    detections = [
      {"name": "manure", "confidence": 0.6, "bbox": [0, 0, 10, 10]},
      {"name": "manure", "confidence": 0.9, "bbox": [1, 1, 11, 11]},
      {"name": "cow", "confidence": 0.99, "bbox": [2, 2, 12, 12]},
    ]

    result = select_best_detection(detections, "manure", 0.5)

    assert result["confidence"] == 0.9
    assert result["bbox"] == [1, 1, 11, 11]

  def test_returns_none_when_below_threshold(self):
    detections = [{"name": "manure", "confidence": 0.4, "bbox": [0, 0, 10, 10]}]

    assert select_best_detection(detections, "manure", 0.5) is None


class TestBuildEventPayload:
  """Tests for API payload creation."""

  def test_builds_payload_with_bbox_dimensions(self):
    detection = {"confidence": 0.75, "bbox": [10, 20, 40, 60]}

    payload = build_event_payload(detection, "cam-a", "edge-1")

    assert payload["camera_id"] == "cam-a"
    assert payload["device_id"] == "edge-1"
    assert payload["confidence"] == 0.75
    assert payload["bounding_box"] == {
      "x": 10,
      "y": 20,
      "width": 30,
      "height": 40,
    }
    assert "timestamp" in payload


class TestEncodeJpeg:
  """Tests for JPEG encoding."""

  def test_encode_jpeg_returns_bytes(self):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result = encode_jpeg(frame, quality=70)

    assert isinstance(result, bytes)
    assert result.startswith(b"\xff\xd8")


class TestPrepareDetectionEvent:
  """Tests for complete event preparation."""

  def test_prepare_detection_event(self):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = [{"name": "manure", "confidence": 0.8, "bbox": [1, 2, 11, 12]}]

    payload, image_bytes, detection = prepare_detection_event(
      detections=detections,
      frame=frame,
      target_name="manure",
      min_confidence=0.5,
      camera_id="cam-a",
      device_id="edge-1",
      jpeg_quality=70,
    )

    assert payload["camera_id"] == "cam-a"
    assert image_bytes is not None
    assert detection == detections[0]

  def test_prepare_detection_event_without_match(self):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    payload, image_bytes, detection = prepare_detection_event(
      detections=[],
      frame=frame,
      target_name="manure",
      min_confidence=0.5,
      camera_id="cam-a",
      device_id="edge-1",
      jpeg_quality=70,
    )

    assert payload is None
    assert image_bytes is None
    assert detection is None
