import cv2
import os
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.videos = self._load_videos()

    def _load_videos(self):
        videos = {}
        sorted_files = sorted(os.listdir(self.video_dir))
        for filename in sorted_files:
            if filename.endswith(".mp4"):
                camera_name = filename.split('_')[1] + '_' + filename.split('_')[2].split('.')[0]
                videos[camera_name] = cv2.VideoCapture(os.path.join(self.video_dir, filename))
        return videos

    def get_frame(self, camera_name, frame_number):
        cap = self.videos.get(camera_name)
        if not cap:
            raise ValueError(f"Camera {camera_name} not found in video directory.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        return frame if ret else None

    def release(self):
        for cap in self.videos.values():
            cap.release()

class DetectionProcessor:
    def __init__(self, video_processor, yolo_detector):
        self.video_processor = video_processor
        self.yolo_detector = yolo_detector

    def process_current_frame_from_all_cameras(self, frame_number):
        detections = {}
        for camera_name in self.video_processor.videos.keys():
            frame = self.video_processor.get_frame(camera_name, frame_number)
            if frame is None:
                print(f"Frame {frame_number} not found in camera {camera_name}.")
                break
            detections[camera_name] = self.yolo_detector.detect_objects(frame)
        return detections

    def process_all_frames(self):
        detections_per_frame = {}
        for camera_name in self.video_processor.videos.keys():
            frame_number = 0
            total_frames = int(self.video_processor.videos[camera_name].get(cv2.CAP_PROP_FRAME_COUNT))
            # total_frames = 1000
            with tqdm(total=total_frames, desc=f"Processing Camera {camera_name}") as pbar:
                while frame_number < total_frames:
                    frame = self.video_processor.get_frame(camera_name, frame_number)
                    if frame is None:
                        break
                    detections = self.yolo_detector.detect_objects(frame)
                    if frame_number not in detections_per_frame:
                        detections_per_frame[frame_number] = []
                    detections_per_frame[frame_number].append({
                        'camera_name': camera_name,
                        'detections': detections
                    })
                    frame_number += 1
                    pbar.update(1)

        return detections_per_frame