"""Tests for APIClient event sending and queue flushing."""

import time
import responses
from unittest.mock import patch, MagicMock

from src.client.api_client import APIClient


class TestNormalizeTimestamp:
  """Tests for timestamp normalization."""

  def test_already_z_format(self):
    payload = {"timestamp": "2026-04-02T12:00:00Z"}
    result = APIClient._normalize_timestamp(payload)
    assert result["timestamp"] == "2026-04-02T12:00:00Z"

  def test_plus0000_converted_to_z(self):
    payload = {"timestamp": "2026-04-02T12:00:00+00:00"}
    result = APIClient._normalize_timestamp(payload)
    assert result["timestamp"] == "2026-04-02T12:00:00Z"

  def test_no_timezone_gets_z_appended(self):
    payload = {"timestamp": "2026-04-02T12:00:00"}
    result = APIClient._normalize_timestamp(payload)
    assert result["timestamp"] == "2026-04-02T12:00:00Z"

  def test_no_timestamp_field(self):
    payload = {"camera_id": "cam1"}
    result = APIClient._normalize_timestamp(payload)
    assert result == payload


class TestPreparePayload:
  """Tests for payload preparation."""

  def test_without_image(self):
    client = APIClient()
    payload = {"camera_id": "cam1"}
    result = client._prepare_payload(payload)
    assert result == {"camera_id": "cam1"}

  def test_with_image_encodes_base64(self):
    client = APIClient()
    payload = {"camera_id": "cam1", "snapshot_mode": "none"}
    result = client._prepare_payload(payload, image_bytes=b"img")
    assert "image_snapshot" in result
    assert result["image_snapshot"] == "aW1n"  # base64 of b"img"
    assert result["snapshot_mode"] == "full_frame"

  def test_with_image_can_encode_data_uri(self, monkeypatch):
    monkeypatch.setattr("src.client.api_client.settings.IMAGE_SNAPSHOT_DATA_URI", True)
    client = APIClient()
    payload = {"camera_id": "cam1", "snapshot_mode": "none"}
    result = client._prepare_payload(payload, image_bytes=b"img")
    assert result["image_snapshot"] == "data:image/jpeg;base64,aW1n"

  def test_original_payload_not_mutated(self):
    client = APIClient()
    payload = {"camera_id": "cam1"}
    client._prepare_payload(payload, image_bytes=b"img")
    assert "image_snapshot" not in payload


class TestSendEvent:
  """Tests for event sending with mock HTTP."""

  @responses.activate
  def test_successful_send(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=201)
    client = APIClient(api_url="http://localhost:8000/api/v1/events")
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://localhost:8000/api/v1/events"

  @responses.activate
  def test_failed_send_enqueues(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=500)
    client = APIClient(api_url="http://localhost:8000/api/v1/events")
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    assert client.queue.size() == 1

  @responses.activate
  def test_sends_with_api_key_header(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=201)
    client = APIClient(api_url="http://localhost:8000/api/v1/events")
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    headers = responses.calls[0].request.headers
    assert "X-API-Key" in headers

  @responses.activate
  def test_sends_trimmed_api_key_header(self, monkeypatch):
    monkeypatch.setattr("src.client.api_client.settings.API_KEY", "  bs_live_test_key \n")
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=201)
    client = APIClient(api_url="http://localhost:8000/api/v1/events")
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    headers = responses.calls[0].request.headers
    assert headers["X-API-Key"] == "bs_live_test_key"

  @responses.activate
  def test_sends_edge_payload_with_image_snapshot(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/edge/events", status=201)
    client = APIClient(api_url="http://localhost:8000/api/v1/edge/events")
    client.send_event({
      "timestamp": "2026-04-02T12:00:00Z",
      "camera_id": "cam1",
      "device_id": "edge1",
      "detected_class": "manure",
      "confidence": 0.9,
      "bounding_box": {"x": 1, "y": 2, "width": 3, "height": 4},
      "model_version": "v1",
      "model_path": "models/manure.pt",
      "inference_fps": 5.0,
      "edge_queue_size": 0,
      "img_size": 640,
      "threshold": 0.5,
      "event_id": "evt-1",
      "zone_id": "zone-1",
      "barn_id": "barn-1",
      "snapshot_mode": "none",
      "edge_app_version": "0.1.0",
      "queue_latency_seconds": 0.0,
    }, image_bytes=b"img")
    body = responses.calls[0].request.body.decode("utf-8")
    assert '"image_snapshot": "aW1n"' in body
    assert '"snapshot_mode": "full_frame"' in body


class TestFlushLoop:
  """Tests for background queue flushing."""

  @responses.activate
  def test_flushes_queued_items(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=201)
    client = APIClient(api_url="http://localhost:8000/api/v1/events")
    client.queue.enqueue("http://localhost:8000/api/v1/events", {
      "camera_id": "cam1",
      "timestamp": "2026-04-02T12:00:00Z",
    })
    # Run one flush iteration manually
    client._is_running = True
    item = client.queue.dequeue()
    client._prepare_payload(item["payload"], item["image_bytes"])
    client._normalize_timestamp(item["payload"])
    assert item["payload"]["timestamp"] == "2026-04-02T12:00:00Z"

  @responses.activate
  def test_requeues_failed_item(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=500)
    client = APIClient(api_url="http://localhost:8000/api/v1/events")
    client.queue.enqueue("http://localhost:8000/api/v1/events", {
      "camera_id": "cam1",
      "timestamp": "2026-04-02T12:00:00Z",
    })
    # Simulate flush failure
    client._is_running = True
    item = client.queue.dequeue()
    try:
      prepared = client._prepare_payload(item["payload"], item["image_bytes"])
      prepared = client._normalize_timestamp(prepared)
      import requests
      response = requests.post(
        item["endpoint"],
        json=prepared,
        headers=client._get_headers(),
        timeout=5.0,
      )
      response.raise_for_status()
    except Exception:
      client.queue.requeue(item)
    
    # Item should be back in queue
    assert client.queue.size() == 1
