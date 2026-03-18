from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union

class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    	env_ignore_empty=True,
      extra="ignore",
  )
    
  STREAM_URL: Union[str, int] = 0
  MODEL_PATH: str = "models/manure.pt"

  FRAME_WIDTH: int = 640
  FRAME_HEIGHT: int = 640

  FRAME_TIMEOUT: int = 60

  MAX_RESTARTS: int = 5
  
  # Event & API Settings
  API_URL: str = "http://localhost:80/api/v1/events"
  API_KEY: str = "" # Must be set in .env
  DEVICE_ID: str = "edge-device-01"
  CAMERA_ID: str = "camera-01"
  COOLDOWN_SECONDS: float = 5.0 # Time to wait before sending another event for the same region/class
  MIN_CONFIDENCE: float = 0.5
  ENABLE_DISPLAY: bool = False
  
  # Edge Hardware Optimizations
  INFERENCE_FPS: float = 5.0  # Limit inference rate to save CPU/GPU
  HALF_PRECISION: bool = False # Use FP16 for faster inference if supported
  IMG_SIZE: int = 640          # Internal inference resolution (reduce to 320 or 416 for speed)

settings = Settings()
