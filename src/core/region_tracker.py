"""Region-based deduplication tracker for detection events.

Prevents sending duplicate images of the same manure spot by
tracking bounding box regions and enforcing per-region cooldowns.
Two detections are considered the same region if their IoU (Intersection
over Union) exceeds the configured threshold.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple


class RegionTracker:
  """Tracks detection regions and enforces cooldowns per region."""

  def __init__(
    self,
    overlap_threshold: float = 0.5,
    cooldown_seconds: float = 5.0,
    ttl_seconds: float = 300.0,
    max_entries: int = 512,
  ):
    self.overlap_threshold = overlap_threshold
    self.cooldown_seconds = cooldown_seconds
    self.ttl_seconds = ttl_seconds
    self.max_entries = max_entries
    # Maps region bbox tuple -> last_send_time
    self._regions: Dict[Tuple[float, float, float, float], float] = {}
    self._lock = threading.Lock()

  @staticmethod
  def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    if inter_area == 0:
      return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
      return 0.0

    return inter_area / union_area

  def _prune(self, now: Optional[float] = None) -> None:
    """Remove stale region entries to bound memory on long-running devices."""
    now = now or time.time()
    expired = [
      key for key, last_seen in self._regions.items()
      if now - last_seen >= self.ttl_seconds
    ]
    for key in expired:
      self._regions.pop(key, None)

    while len(self._regions) > self.max_entries:
      oldest_key = min(self._regions, key=self._regions.get)
      self._regions.pop(oldest_key, None)

  def _find_matching_region(
    self,
    bbox: List[float],
  ) -> Optional[Tuple[float, float, float, float]]:
    """Find an existing region that overlaps with this bbox.

    Returns the matching region key if found, or None.
    """
    best_key = None
    best_iou = 0.0
    for region_key in self._regions:
      region_bbox = list(region_key)
      iou = self._iou(bbox, region_bbox)
      if iou > best_iou:
        best_iou = iou
        best_key = region_key
    if best_iou >= self.overlap_threshold:
      return best_key
    return None

  def should_send(self, bbox: List[float]) -> bool:
    """Check if a detection at this bbox should trigger a send.

    Returns True if no matching region exists or the cooldown has expired.
    """
    with self._lock:
      now = time.time()
      self._prune(now)
      match = self._find_matching_region(bbox)
      if match is None:
        return True
      elapsed = now - self._regions[match]
      return elapsed >= self.cooldown_seconds

  def mark_sent(self, bbox: List[float]) -> None:
    """Record that an image was sent for this region.

    If a matching region exists, updates its timestamp.
    Otherwise creates a new region entry.
    """
    with self._lock:
      now = time.time()
      self._prune(now)
      match = self._find_matching_region(bbox)
      key = match if match is not None else tuple(round(v, 1) for v in bbox)
      self._regions[key] = now

  def check_and_mark(self, bbox: List[float]) -> bool:
    """Atomic check-and-mark: returns True if should send, then marks.

    This is the primary method used by the inference loop.
    """
    with self._lock:
      now = time.time()
      self._prune(now)
      match = self._find_matching_region(bbox)
      if match is not None and now - self._regions[match] < self.cooldown_seconds:
        return False
      key = match if match is not None else tuple(round(v, 1) for v in bbox)
      self._regions[key] = now
      return True

  def reset(self) -> None:
    """Clear all tracked regions."""
    with self._lock:
      self._regions.clear()
