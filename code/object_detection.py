import cv2
from ultralytics import YOLO
import torch


class YOLODetector:
    def __init__(self, yolo_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(yolo_model_path)


    def detect_objects(self, frame):
        with torch.no_grad():
            # self.model = self.model.to(self.device)
            self.model = self.model.cuda()
            results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            for detection in result.boxes:
                bbox = detection.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
                class_id = int(detection.cls[0])   # Class ID
                confidence = float(detection.conf[0])  # Confidence score
                if confidence < 0.3:
                    continue
                detections.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'confidence': confidence
                })
        
        return detections