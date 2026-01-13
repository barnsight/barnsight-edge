from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union

class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    	env_ignore_empty=True,
      extra="ignore",
  )
    
  STREAM_URL: Union[str, int] = 0
  MODEL_PATH: str = "models/yolo11n.pt"

  FRAME_WIDTH: int = 640
  FRAME_HEIGHT: int = 640

  FRAME_TIMEOUT: int = 60

  MAX_RESTARTS: int = 5

  API_HOST: str = "0.0.0.0"
  API_PORT: int = 8000
  RELOAD: bool = False

settings = Settings()
