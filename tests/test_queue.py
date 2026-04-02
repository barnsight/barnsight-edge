"""Tests for the in-memory DetectionQueue."""

from src.core.queue import DetectionQueue


class TestEnqueueDequeue:
  """Tests for basic enqueue/dequeue operations."""

  def test_enqueue_and_dequeue(self):
    queue = DetectionQueue()
    queue.enqueue("http://api/events", {"key": "val"})
    item = queue.dequeue()
    assert item is not None
    assert item["endpoint"] == "http://api/events"
    assert item["payload"] == {"key": "val"}
    assert item["image_bytes"] is None

  def test_dequeue_empty_returns_none(self):
    queue = DetectionQueue()
    assert queue.dequeue() is None

  def test_fifo_order(self):
    queue = DetectionQueue()
    queue.enqueue("url", {"id": 1})
    queue.enqueue("url", {"id": 2})
    queue.enqueue("url", {"id": 3})
    assert queue.dequeue()["payload"]["id"] == 1
    assert queue.dequeue()["payload"]["id"] == 2
    assert queue.dequeue()["payload"]["id"] == 3

  def test_with_image_bytes(self):
    queue = DetectionQueue()
    data = b"\xff\xd8\xff\xe0"
    queue.enqueue("url", {"cam": "a"}, image_bytes=data)
    item = queue.dequeue()
    assert item["image_bytes"] == data

  def test_size_increments(self):
    queue = DetectionQueue()
    assert queue.size() == 0
    queue.enqueue("url", {"id": 1})
    assert queue.size() == 1
    queue.enqueue("url", {"id": 2})
    assert queue.size() == 2

  def test_size_decrements_on_dequeue(self):
    queue = DetectionQueue()
    queue.enqueue("url", {"id": 1})
    queue.enqueue("url", {"id": 2})
    queue.dequeue()
    assert queue.size() == 1


class TestRequeue:
  """Tests for requeue (failed item retry) logic."""

  def test_requeue_puts_item_at_back(self):
    queue = DetectionQueue()
    queue.enqueue("url", {"id": 1})
    queue.enqueue("url", {"id": 2})
    item = queue.dequeue()
    assert item["payload"]["id"] == 1
    queue.requeue(item)
    # Item 1 should now be after item 2
    assert queue.dequeue()["payload"]["id"] == 2
    assert queue.dequeue()["payload"]["id"] == 1

  def test_requeue_preserves_data(self):
    queue = DetectionQueue()
    queue.enqueue("url", {"key": "val"}, image_bytes=b"data")
    item = queue.dequeue()
    queue.requeue(item)
    restored = queue.dequeue()
    assert restored["payload"] == {"key": "val"}
    assert restored["image_bytes"] == b"data"
    assert restored["endpoint"] == "url"


class TestMaxSize:
  """Tests for bounded queue behavior."""

  def test_maxsize_drops_oldest(self):
    queue = DetectionQueue(maxsize=2)
    queue.enqueue("url", {"id": 1})
    queue.enqueue("url", {"id": 2})
    queue.enqueue("url", {"id": 3})
    # Oldest item should be dropped
    assert queue.size() == 2
    assert queue.dequeue()["payload"]["id"] == 2
    assert queue.dequeue()["payload"]["id"] == 3
