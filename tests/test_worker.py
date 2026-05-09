"""Tests for inference worker event handling."""

import numpy as np

from src.config import settings
from src.core.region_tracker import RegionTracker
from src.inference import worker as worker_module
from src.inference.worker import InferenceWorker


class StubAPIClient:
  """Collect submitted events without network calls."""

  def __init__(self):
    self.events = []

  def submit_event(self, payload, image_bytes):
    self.events.append((payload, image_bytes))


def build_worker() -> tuple[InferenceWorker, StubAPIClient]:
  api_client = StubAPIClient()
  inference_worker = InferenceWorker()
  inference_worker.region_tracker = RegionTracker(cooldown_seconds=999.0)
  inference_worker.api_client = api_client
  return inference_worker, api_client


class TestHandleDetections:
  """Tests for per-frame detection submission behavior."""

  def test_submits_each_matching_manure_detection(self, monkeypatch):
    monkeypatch.setattr(settings, "MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(settings, "MAX_DETECTIONS_PER_FRAME", 20)
    inference_worker, api_client = build_worker()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = [
      {"name": "manure", "confidence": 0.8, "bbox": [1, 2, 11, 12]},
      {"name": "cow", "confidence": 0.99, "bbox": [3, 4, 13, 14]},
      {"name": "Manure", "confidence": 0.7, "bbox": [20, 20, 30, 30]},
    ]

    inference_worker._handle_detections(detections, frame, current_time=10.0)

    assert len(api_client.events) == 2
    assert [event[0]["confidence"] for event in api_client.events] == [0.8, 0.7]
    assert api_client.events[0][1] is api_client.events[1][1]

  def test_skips_jpeg_encoding_when_regions_are_duplicates(self, monkeypatch):
    monkeypatch.setattr(settings, "MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(settings, "MAX_DETECTIONS_PER_FRAME", 20)
    inference_worker, api_client = build_worker()
    inference_worker.region_tracker.mark_sent([1, 2, 11, 12])
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = [{"name": "manure", "confidence": 0.8, "bbox": [1, 2, 11, 12]}]

    def fail_encode(*args, **kwargs):
      raise AssertionError("JPEG encoding should not run for duplicate regions")

    monkeypatch.setattr(worker_module, "encode_jpeg", fail_encode)

    inference_worker._handle_detections(detections, frame, current_time=10.0)

    assert api_client.events == []

  def test_limits_events_per_frame(self, monkeypatch):
    monkeypatch.setattr(settings, "MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(settings, "MAX_DETECTIONS_PER_FRAME", 1)
    inference_worker, api_client = build_worker()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = [
      {"name": "manure", "confidence": 0.8, "bbox": [1, 2, 11, 12]},
      {"name": "manure", "confidence": 0.7, "bbox": [20, 20, 30, 30]},
    ]

    inference_worker._handle_detections(detections, frame, current_time=10.0)

    assert len(api_client.events) == 1
    assert api_client.events[0][0]["confidence"] == 0.8
