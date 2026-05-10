"""Application settings loaded from .env file.

Uses pydantic-settings for type-validated environment variable
parsing with automatic .env file loading.
"""

from typing import Literal, Union

from pydantic import field_validator
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
  STREAM_FPS: int = 30
  STREAM_RECONNECT_INITIAL_DELAY: float = 0.1
  STREAM_RECONNECT_MAX_DELAY: float = 5.0
  CAMERA_STALE_SECONDS: float = 10.0
  CAMERA_FROZEN_SECONDS: float = 30.0

  # Model settings
  MODEL_PATH: str = "models/manure.pt"

  # Inference settings
  INFERENCE_FPS: float = 5.0
  DETECTION_CONFIDENCE: float = 0.25
  HALF_PRECISION: bool = False
  IMG_SIZE: int = 640

  # Event & API settings
  API_URL: str = "http://localhost:8000/api/v1/edge/events"
  API_KEY: str = ""  # Must be set in .env
  DEVICE_ID: str = "edge-device-01"
  CAMERA_ID: str = "camera-01"
  BARN_ID: str = ""
  ZONE_ID: str = ""
  API_CONNECT_TIMEOUT_SECONDS: float = 3.0
  API_TIMEOUT_SECONDS: float = 10.0
  API_MAX_RETRIES: int = 2
  API_BACKOFF_SECONDS: float = 0.5
  API_VERIFY_TLS: bool = True
  REQUIRE_HTTPS: bool = False
  EVENT_SEND_WORKERS: int = 2
  QUEUE_MAX_SIZE: int = 1000
  QUEUE_BACKEND: Literal["memory", "sqlite"] = "memory"
  QUEUE_DB_PATH: str = "data/events_queue.sqlite3"
  QUEUE_MAX_RETRY_COUNT: int = 0
  QUEUE_STORE_IMAGES: bool = False
  MAX_IMAGE_BYTES: int = 750_000
  COOLDOWN_SECONDS: float = 1.0
  MIN_CONFIDENCE: float = 0.5
  MAX_DETECTIONS_PER_FRAME: int = 20
  SNAPSHOT_MODE: str = "none"
  EDGE_APP_VERSION: str = "0.1.0"
  MODEL_VERSION: str = ""

  # Display settings
  ENABLE_DISPLAY: bool = False

  # Image encoding settings
  JPEG_QUALITY: int = 70
  IMAGE_SNAPSHOT_DATA_URI: bool = False

  # Region-based deduplication settings
  # Cooldown per spatial region before sending another image
  IMAGE_COOLDOWN_SECONDS: float = 5.0
  # IoU threshold: two bboxes with overlap >= this are considered the same region
  REGION_OVERLAP_THRESHOLD: float = 0.5
  REGION_TTL_SECONDS: float = 300.0
  REGION_MAX_ENTRIES: int = 512

  # Logging
  LOG_LEVEL: str = "INFO"

  @field_validator(
    "INFERENCE_FPS",
    "COOLDOWN_SECONDS",
    "IMAGE_COOLDOWN_SECONDS",
    "API_CONNECT_TIMEOUT_SECONDS",
    "API_TIMEOUT_SECONDS",
    "API_BACKOFF_SECONDS",
    "STREAM_RECONNECT_INITIAL_DELAY",
    "STREAM_RECONNECT_MAX_DELAY",
    "CAMERA_STALE_SECONDS",
    "CAMERA_FROZEN_SECONDS",
  )
  @classmethod
  def _positive_float(cls, value: float) -> float:
    if value <= 0:
      raise ValueError("value must be greater than 0")
    return value

  @field_validator(
    "DETECTION_CONFIDENCE",
    "MIN_CONFIDENCE",
    "REGION_OVERLAP_THRESHOLD",
  )
  @classmethod
  def _unit_interval(cls, value: float) -> float:
    if not 0.0 <= value <= 1.0:
      raise ValueError("value must be between 0.0 and 1.0")
    return value

  @field_validator("JPEG_QUALITY")
  @classmethod
  def _jpeg_quality(cls, value: int) -> int:
    if not 1 <= value <= 100:
      raise ValueError("JPEG_QUALITY must be between 1 and 100")
    return value

  @field_validator(
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "IMG_SIZE",
    "STREAM_FPS",
    "EVENT_SEND_WORKERS",
    "QUEUE_MAX_SIZE",
    "MAX_IMAGE_BYTES",
    "MAX_DETECTIONS_PER_FRAME",
    "REGION_MAX_ENTRIES",
  )
  @classmethod
  def _positive_int(cls, value: int) -> int:
    if value <= 0:
      raise ValueError("value must be greater than 0")
    return value

  @field_validator("API_MAX_RETRIES", "QUEUE_MAX_RETRY_COUNT")
  @classmethod
  def _non_negative_int(cls, value: int) -> int:
    if value < 0:
      raise ValueError("value must be zero or greater")
    return value

  @field_validator("LOG_LEVEL")
  @classmethod
  def _log_level(cls, value: str) -> str:
    level = value.upper()
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
      raise ValueError(f"LOG_LEVEL must be one of: {', '.join(sorted(allowed))}")
    return level


settings = Settings()
