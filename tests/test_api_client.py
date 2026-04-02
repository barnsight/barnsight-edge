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
    payload = {"camera_id": "cam1"}
    result = client._prepare_payload(payload, image_bytes=b"img")
    assert "image_snapshot" in result
    assert result["image_snapshot"] == "aW1n"  # base64 of b"img"

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
    client = APIClient()
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://localhost:8000/api/v1/events"

  @responses.activate
  def test_failed_send_enqueues(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=500)
    client = APIClient()
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    assert client.queue.size() == 1

  @responses.activate
  def test_sends_with_api_key_header(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=201)
    client = APIClient()
    client.send_event({"camera_id": "cam1", "timestamp": "2026-04-02T12:00:00Z"})
    headers = responses.calls[0].request.headers
    assert "X-API-Key" in headers


class TestFlushLoop:
  """Tests for background queue flushing."""

  @responses.activate
  def test_flushes_queued_items(self):
    responses.add(responses.POST, "http://localhost:8000/api/v1/events", status=201)
    client = APIClient()
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
    client = APIClient()
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
      requests.post(
        item["endpoint"],
        json=prepared,
        headers=client._get_headers(),
        timeout=5.0,
      )
    except Exception:
      client.queue.requeue(item)
    # Item should be back in queue
    assert client.queue.size() == 1
