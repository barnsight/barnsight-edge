from core.camera import Camera
from core.logger import logger
from dotenv import load_dotenv
import time
import cv2
import os

load_dotenv()

WIDTH, HEIGHT = os.getenv("WIDTH"), os.getenv("HEIGHT")
FRAME_TIMEOUT = os.getenv("FRAME_TIMEOUT")

if __name__ == "__main__":
  camera = Camera(os.getenv("RTSP_URL"), width=WIDTH, height=HEIGHT)
  camera.start()

  start_time = time.time()
  last_frame_time = time.time()

  try:
    start_time = time.time()

    while True:
      ret, frame = camera.read()
      if not ret:
        if time.time() - last_frame_time > FRAME_TIMEOUT:
          logger.error("No frames received for too long — stream likely dead")
          break
        time.sleep(0.05)
        continue

      last_frame_time = time.time()

      cv2.imshow("Camera", cv2.resize(frame, (WIDTH, HEIGHT)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
    camera.stop()
    cv2.destroyAllWindows()
  
  except cv2.error as e:
    logger.error("An error occured while proccessing frames.", e)