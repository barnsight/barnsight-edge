from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union

class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    	env_ignore_empty=True,
      extra="ignore",
  )
    
  STREAM_URL: Union[str, int] = 0
  
  FRAME_WIDTH: int = 1280
  FRAME_HEIGHT: int = 720

  FRAME_TIMEOUT: int = 60

  MAX_RESTARTS: int = 5

settings = Settings()