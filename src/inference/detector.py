from typing import List, Literal
import numpy as np
import torch
import cv2
import os

from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.core.logger import logger

class Detector:
  def __init__(
      self,
      model_path: str = "models/",
      device: Literal["auto", "cpu", "cuda"] = "auto",
      confidence: float = 0.25,
      iou: float = 0.7
    ):
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"File not found: {model_path}")

    if not 0.0 <= confidence <= 1.0:
      raise ValueError("Confidence must be 0.0-1.0")

    if device == "auto":
      device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
      logger.warning("Cuda not available, falling back to CPU (slower)")
      device = "cpu"

    # Load model
    self.model = YOLO(model_path)
    self.model.to(device)

    # Setting up configuration
    self.model_path = model_path
    self.confidence = confidence
    self.iou_threshold = iou

  @property
  def list_models(self) -> List[str]:
    files = os.listdir(os.path.dirname(self.model_path))
    return [f for f in files if os.path.isfile(self.model_path + "/" + f)]
  
  def predict(self, frame: cv2.Mat, verbose: bool = False) -> np.ndarray:
    results: Results = self.model.predict(
      frame,
      conf=self.confidence,
      iou=self.iou_threshold,
      verbose=verbose
    )[0]

    return results.plot()