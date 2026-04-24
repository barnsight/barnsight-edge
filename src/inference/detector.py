"""YOLO-based object detector for manure detection.

Wraps Ultralytics YOLO model with device auto-detection,
confidence thresholding, and structured output formatting.
"""

import os
from typing import Dict, List, Literal, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.core.logger import logger


class Detector:
  """YOLO model wrapper for real-time object detection."""

  def __init__(
    self,
    model_path: str = "models/",
    device: Literal["auto", "cpu", "cuda"] = "auto",
    confidence: float = 0.25,
    iou: float = 0.7,
    half_precision: bool = False,
    img_size: int = 640,
  ):
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0.0 <= confidence <= 1.0:
      raise ValueError("Confidence must be between 0.0 and 1.0")

    # Auto-select device: prefer CUDA, fall back to CPU
    if device == "auto":
      device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
      logger.warning("CUDA not available, falling back to CPU (slower)")
      device = "cpu"

    self.model = YOLO(model_path)
    self.model.to(device)
    self.model_path = model_path
    self.confidence = confidence
    self.iou_threshold = iou
    # FP16 not supported on CPU
    self.half_precision = half_precision and device != "cpu"
    self.img_size = img_size

  @property
  def list_models(self) -> List[str]:
    """List all model files in the model directory."""
    model_dir = os.path.dirname(self.model_path) or "."
    return [
      f for f in os.listdir(model_dir)
      if os.path.isfile(os.path.join(model_dir, f))
    ]

  def predict(
    self,
    frame: np.ndarray,
    verbose: bool = False,
    annotate: bool = True,
  ) -> Tuple[np.ndarray, List[Dict]]:
    """Run inference on a single frame.

    Args:
      frame: BGR image as numpy array.
      verbose: Print YOLO inference stats.
      annotate: Draw result overlays. Disable on headless edge devices to save CPU.

    Returns:
      Tuple of (annotated frame, list of detection dicts).
    """
    results: Results = self.model.predict(
      frame,
      conf=self.confidence,
      iou=self.iou_threshold,
      imgsz=self.img_size,
      half=self.half_precision,
      verbose=verbose,
    )[0]

    detections: List[Dict] = []
    if results.boxes:
      for box in results.boxes:
        b = box.xyxy[0].tolist()
        c = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        name = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
        detections.append({
          "bbox": b,
          "confidence": c,
          "class_id": cls_id,
          "name": name,
        })

    output_frame = results.plot() if annotate else frame
    return output_frame, detections
