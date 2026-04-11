from ultralytics import YOLO
import sys
try:
    model = YOLO("models/manure-2.pt")
    print(model.names)
except Exception as e:
    print("Error:", e)
