from src.inference.detector import Detector
import cv2
import numpy as np
from src.config import settings

detector = Detector(model_path=settings.MODEL_PATH)
img = np.zeros((640, 640, 3), dtype=np.uint8)
annotated, detections = detector.predict(img)
print("annotated type:", type(annotated))
ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY])
print("ok:", ok)
image_bytes = buffer.tobytes() if ok else None
print("image_bytes is None:", image_bytes is None)
