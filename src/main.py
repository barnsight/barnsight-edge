from core.stream_handler import SteamHandler
import time
import cv2

from config import settings
from core.logger import logger

if __name__ == "__main__":
  camera = SteamHandler(settings.STREAM_URL, width=settings.FRAME_WIDTH, height=settings.FRAME_HEIGHT)
  camera.start()

  start_time = time.time()
  last_frame_time = time.time()

  try:
    start_time = time.time()

    while True:
      ret, frame = camera.read()
      if not ret:
        if time.time() - last_frame_time > settings.FRAME_TIMEOUT:
          logger.error("No frames received for too long — stream likely dead")
          break
        time.sleep(0.05)
        continue

      last_frame_time = time.time()

      cv2.imshow("Camera", cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
    camera.stop()
    cv2.destroyAllWindows()
  
  except cv2.error as e:
    logger.error("An error occured while proccessing frames.", e)