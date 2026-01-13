import uvicorn
from src.config import settings

if __name__ == "__main__":
  uvicorn.run(
    "src.frontend.server:app",
    host=settings.API_HOST,
    port=settings.API_PORT,
    reload=settings.RELOAD,
  )