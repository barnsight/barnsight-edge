"""Tests for detection event helper functions."""

import numpy as np

from src.core.events import (
  build_event_payload,
  encode_jpeg,
  prepare_detection_event,
  prepare_detection_events,
  select_best_detection,
  select_target_detections,
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


class TestSelectTargetDetections:
  """Tests for selecting every matching target detection."""

  def test_selects_all_matching_targets_by_confidence(self):
    detections = [
      {"name": "manure", "confidence": 0.6, "bbox": [0, 0, 10, 10]},
      {"name": "Manure", "confidence": 0.9, "bbox": [1, 1, 11, 11]},
      {"name": "cow", "confidence": 0.99, "bbox": [2, 2, 12, 12]},
      {"name": "manure", "confidence": 0.4, "bbox": [3, 3, 13, 13]},
    ]

    result = select_target_detections(detections, "manure", 0.5)

    assert [detection["confidence"] for detection in result] == [0.9, 0.6]


class TestBuildEventPayload:
  """Tests for API payload creation."""

  def test_builds_payload_with_bbox_dimensions(self):
    detection = {"name": "manure", "confidence": 0.75, "bbox": [10, 20, 40, 60]}

    payload = build_event_payload(
      detection,
      "cam-a",
      "edge-1",
      barn_id="barn-1",
      zone_id="zone-2",
      model_version="v1",
      model_path="models/manure.pt",
      inference_fps=5.0,
      img_size=640,
      threshold=0.5,
      edge_app_version="0.1.0",
      snapshot_mode="none",
    )

    assert payload["camera_id"] == "cam-a"
    assert payload["device_id"] == "edge-1"
    assert payload["detected_class"] == "manure"
    assert payload["confidence"] == 0.75
    assert payload["bounding_box"] == {
      "x": 10,
      "y": 20,
      "width": 30,
      "height": 40,
    }
    assert payload["barn_id"] == "barn-1"
    assert payload["zone_id"] == "zone-2"
    assert payload["model_version"] == "v1"
    assert payload["model_path"] == "models/manure.pt"
    assert payload["inference_fps"] == 5.0
    assert payload["edge_queue_size"] == 0
    assert payload["img_size"] == 640
    assert payload["threshold"] == 0.5
    assert payload["snapshot_mode"] == "none"
    assert payload["edge_app_version"] == "0.1.0"
    assert payload["queue_latency_seconds"] == 0.0
    assert "event_id" in payload
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


class TestPrepareDetectionEvents:
  """Tests for preparing one event per matching detection."""

  def test_prepare_detection_events(self):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = [
      {"name": "manure", "confidence": 0.8, "bbox": [1, 2, 11, 12]},
      {"name": "manure", "confidence": 0.7, "bbox": [3, 4, 13, 14]},
    ]

    events, image_bytes = prepare_detection_events(
      detections=detections,
      frame=frame,
      target_name="manure",
      min_confidence=0.5,
      camera_id="cam-a",
      device_id="edge-1",
      jpeg_quality=70,
    )

    assert len(events) == 2
    assert image_bytes is not None
    assert events[0][0]["confidence"] == 0.8
    assert events[1][0]["confidence"] == 0.7

  def test_prepare_detection_events_without_match(self):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    events, image_bytes = prepare_detection_events(
      detections=[],
      frame=frame,
      target_name="manure",
      min_confidence=0.5,
      camera_id="cam-a",
      device_id="edge-1",
      jpeg_quality=70,
    )

    assert events == []
    assert image_bytes is None
