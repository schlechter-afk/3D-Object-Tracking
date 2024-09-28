import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, yolo_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(yolo_model_path).to(self.device)

    def detect_objects(self, frame):
        print(f"YOLO model loaded from {yolo_model_path} on device {self.device}.")
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            for detection in result.boxes:
                bbox = detection.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
                class_id = int(detection.cls[0])   # Class ID
                confidence = float(detection.conf[0])  # Confidence score
                
                detections.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'confidence': confidence
                })
        
        return detections