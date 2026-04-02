"""Application settings loaded from .env file.

Uses pydantic-settings for type-validated environment variable
parsing with automatic .env file loading.
"""

from typing import Union

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  """BarnSight Edge configuration."""

  model_config = SettingsConfigDict(
    env_file=".env",
    env_ignore_empty=True,
    extra="ignore",
  )

  # Camera / stream settings
  STREAM_URL: Union[str, int] = 0
  FRAME_WIDTH: int = 640
  FRAME_HEIGHT: int = 640

  # Model settings
  MODEL_PATH: str = "models/manure.pt"

  # Inference settings
  INFERENCE_FPS: float = 5.0
  HALF_PRECISION: bool = False
  IMG_SIZE: int = 640

  # Event & API settings
  API_URL: str = "http://localhost:8000/api/v1/events"
  API_KEY: str = ""  # Must be set in .env
  DEVICE_ID: str = "edge-device-01"
  CAMERA_ID: str = "camera-01"
  COOLDOWN_SECONDS: float = 1.0
  MIN_CONFIDENCE: float = 0.5

  # Display settings
  ENABLE_DISPLAY: bool = False

  # Image encoding settings
  JPEG_QUALITY: int = 70

  # Region-based deduplication settings
  # Cooldown per spatial region before sending another image
  IMAGE_COOLDOWN_SECONDS: float = 5.0
  # IoU threshold: two bboxes with overlap >= this are considered the same region
  REGION_OVERLAP_THRESHOLD: float = 0.5


settings = Settings()
