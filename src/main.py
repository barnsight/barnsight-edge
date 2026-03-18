import time
import signal
import sys
import datetime
import cv2
from typing import Optional

from src.config import settings
from src.core.logger import logger
from src.core.stream_handler import StreamHandler
from src.inference.detector import Detector
from src.client.api_client import APIClient

class InferenceWorker:
    def __init__(self):
        self.camera: Optional[StreamHandler] = None
        self.detector: Optional[Detector] = None
        self.api_client: Optional[APIClient] = None
        self._is_running = False
        self.last_detection_time = 0.0

    def setup(self):
        try:
            self.camera = StreamHandler(
                settings.STREAM_URL,
                width=settings.FRAME_WIDTH,
                height=settings.FRAME_HEIGHT,
            )
            self.camera.start()
            logger.info("[+] Camera initialized")
        except Exception as exc:
            logger.error(f"[x] Failed to initialize camera: {exc}")
            sys.exit(1)

        try:
            self.detector = Detector(
                model_path=settings.MODEL_PATH,
                half_precision=settings.HALF_PRECISION,
                img_size=settings.IMG_SIZE
            )
            logger.info(f"[+] Detector loaded from {settings.MODEL_PATH} (imgsz={settings.IMG_SIZE}, fp16={settings.HALF_PRECISION})")
        except Exception as exc:
            logger.error(f"[x] Failed to load detector: {exc}")
            sys.exit(1)
            
        self.api_client = APIClient()
        self.api_client.start()

    def stop(self, *args):
        logger.info("[*] Stopping inference worker...")
        self._is_running = False
        if self.camera:
            self.camera.stop()
        if self.api_client:
            self.api_client.stop()
        if settings.ENABLE_DISPLAY:
            cv2.destroyAllWindows()
        sys.exit(0)

    def run(self):
        self._is_running = True
        logger.info(f"[*] Inference worker started (Target FPS: {settings.INFERENCE_FPS})")
        
        frame_interval = 1.0 / settings.INFERENCE_FPS if settings.INFERENCE_FPS > 0 else 0
        last_inference_time = 0.0
        
        while self._is_running:
            if not self.camera or not self.detector:
                time.sleep(0.1)
                continue

            # Always pull the latest frame to keep the OpenCV buffer empty
            ret, frame = self.camera.read()
            if not ret or frame is None:
                # Camera not ready / RTSP down
                time.sleep(0.05)
                continue

            current_time = time.time()
            
            # Throttle inference to save edge resources
            if (current_time - last_inference_time) < frame_interval:
                # Still show display if enabled, just without new detections
                if settings.ENABLE_DISPLAY:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quitting display window")
                        self.stop()
                time.sleep(0.01) # tiny sleep to avoid CPU spinning while dropping frames
                continue

            last_inference_time = current_time

            try:
                annotated, detections = self.detector.predict(frame)

                
                if settings.ENABLE_DISPLAY:
                    cv2.imshow("BarnSight Edge - Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quitting display window")
                        self.stop()
                
                # Check cooldown to prevent redundant event spamming
                if detections and (current_time - self.last_detection_time > settings.COOLDOWN_SECONDS):
                    # Filter for highest confidence detection above minimum threshold
                    best_detection = None
                    for det in detections:
                        if det["confidence"] >= settings.MIN_CONFIDENCE:
                            if not best_detection or det["confidence"] > best_detection["confidence"]:
                                best_detection = det
                                
                    if best_detection:
                        self.last_detection_time = current_time
                        
                        # Compress image for transmission
                        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        image_bytes = buffer.tobytes() if ok else None
                        
                        # Build payload according to API schema
                        bbox = best_detection["bbox"]
                        payload = {
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "camera_id": settings.CAMERA_ID,
                            "device_id": settings.DEVICE_ID,
                            "confidence": best_detection["confidence"],
                            "bounding_box": {
                                "x": bbox[0],
                                "y": bbox[1],
                                "width": bbox[2] - bbox[0],
                                "height": bbox[3] - bbox[1]
                            }
                        }
                        
                        logger.info(f"Detected event. Confidence: {payload['confidence']:.2f}")
                        
                        # We use threading to prevent blocking the inference loop
                        import threading
                        threading.Thread(
                            target=self.api_client.send_event,
                            args=(payload, image_bytes),
                            daemon=True
                        ).start()
            except Exception as exc:
                logger.error(f"[x] Detector error: {exc}")

            # Small delay to prevent overloading CPU
            time.sleep(0.01)

if __name__ == "__main__":
    worker = InferenceWorker()
    
    # Handle termination signals
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    
    worker.setup()
    worker.run()
