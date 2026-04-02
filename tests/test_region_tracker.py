"""Tests for the RegionTracker deduplication system."""

import time
from src.core.region_tracker import RegionTracker


class TestIoU:
  """Tests for bounding box IoU calculation."""

  def test_identical_boxes(self):
    box = [100, 100, 200, 200]
    assert RegionTracker._iou(box, box) == 1.0

  def test_non_overlapping_boxes(self):
    box_a = [0, 0, 50, 50]
    box_b = [100, 100, 150, 150]
    assert RegionTracker._iou(box_a, box_b) == 0.0

  def test_partial_overlap(self):
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]
    iou = RegionTracker._iou(box_a, box_b)
    assert 0.1 < iou < 0.3

  def test_one_inside_other(self):
    box_a = [0, 0, 200, 200]
    box_b = [50, 50, 150, 150]
    iou = RegionTracker._iou(box_a, box_b)
    assert iou > 0.2

  def test_adjacent_boxes(self):
    box_a = [0, 0, 100, 100]
    box_b = [100, 0, 200, 100]
    assert RegionTracker._iou(box_a, box_b) == 0.0

  def test_zero_area_box(self):
    box_a = [0, 0, 0, 0]
    box_b = [10, 10, 20, 20]
    assert RegionTracker._iou(box_a, box_b) == 0.0


class TestShouldSend:
  """Tests for should_send logic."""

  def test_new_region_always_sends(self):
    tracker = RegionTracker()
    assert tracker.should_send([100, 100, 200, 200]) is True

  def test_same_region_within_cooldown_blocked(self):
    tracker = RegionTracker(cooldown_seconds=5.0)
    tracker.mark_sent([100, 100, 200, 200])
    assert tracker.should_send([100, 100, 200, 200]) is False

  def test_same_region_after_cooldown_allowed(self):
    tracker = RegionTracker(cooldown_seconds=0.1)
    tracker.mark_sent([100, 100, 200, 200])
    time.sleep(0.15)
    assert tracker.should_send([100, 100, 200, 200]) is True

  def test_different_region_allowed(self):
    tracker = RegionTracker()
    tracker.mark_sent([0, 0, 50, 50])
    assert tracker.should_send([300, 300, 350, 350]) is True

  def test_overlapping_below_threshold_allowed(self):
    tracker = RegionTracker(overlap_threshold=0.8)
    tracker.mark_sent([0, 0, 100, 100])
    # Partial overlap well below 0.8 threshold
    assert tracker.should_send([50, 50, 150, 150]) is True

  def test_overlapping_above_threshold_blocked(self):
    tracker = RegionTracker(overlap_threshold=0.1)
    tracker.mark_sent([0, 0, 100, 100])
    # Same partial overlap but threshold is low
    assert tracker.should_send([50, 50, 150, 150]) is False


class TestCheckAndMark:
  """Tests for the atomic check_and_mark method."""

  def test_first_detection_sends(self):
    tracker = RegionTracker()
    assert tracker.check_and_mark([100, 100, 200, 200]) is True

  def test_duplicate_detection_blocked(self):
    tracker = RegionTracker()
    tracker.check_and_mark([100, 100, 200, 200])
    assert tracker.check_and_mark([100, 100, 200, 200]) is False

  def test_different_region_sends(self):
    tracker = RegionTracker()
    tracker.check_and_mark([0, 0, 50, 50])
    assert tracker.check_and_mark([300, 300, 350, 350]) is True

  def test_multiple_regions_tracked(self):
    tracker = RegionTracker()
    tracker.check_and_mark([0, 0, 50, 50])
    tracker.check_and_mark([100, 100, 150, 150])
    tracker.check_and_mark([200, 200, 250, 250])
    # All three regions should be blocked now
    assert tracker.check_and_mark([0, 0, 50, 50]) is False
    assert tracker.check_and_mark([100, 100, 150, 150]) is False
    assert tracker.check_and_mark([200, 200, 250, 250]) is False

  def test_cooldown_expiry_allows_resend(self):
    tracker = RegionTracker(cooldown_seconds=0.1)
    tracker.check_and_mark([100, 100, 200, 200])
    time.sleep(0.15)
    assert tracker.check_and_mark([100, 100, 200, 200]) is True

  def test_slightly_shifted_bbox_same_region(self):
    tracker = RegionTracker(overlap_threshold=0.5)
    tracker.check_and_mark([100, 100, 200, 200])
    # Slightly shifted — high IoU, should be blocked
    assert tracker.check_and_mark([105, 105, 205, 205]) is False

  def test_far_shifted_bbox_different_region(self):
    tracker = RegionTracker(overlap_threshold=0.5)
    tracker.check_and_mark([0, 0, 50, 50])
    # Completely different area
    assert tracker.check_and_mark([400, 400, 450, 450]) is True


class TestReset:
  """Tests for tracker reset."""

  def test_reset_clears_all_regions(self):
    tracker = RegionTracker()
    tracker.check_and_mark([100, 100, 200, 200])
    tracker.check_and_mark([0, 0, 50, 50])
    tracker.reset()
    # After reset, all regions should be allowed again
    assert tracker.check_and_mark([100, 100, 200, 200]) is True
    assert tracker.check_and_mark([0, 0, 50, 50]) is True

  def test_reset_allows_immediate_resend(self):
    tracker = RegionTracker(cooldown_seconds=999)
    tracker.check_and_mark([100, 100, 200, 200])
    tracker.reset()
    assert tracker.check_and_mark([100, 100, 200, 200]) is True
