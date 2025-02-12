from ultralytics import YOLO
from PIL import Image
import cv2
model = YOLO("yolov8m.pt")
 # accepts all formats image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0", show=True)
