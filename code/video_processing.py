import cv2
import os
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, video_dir, camera_calibration):
        self.video_dir = video_dir
        self.calib = camera_calibration
        self.videos = self._load_videos()

        # Pre-load camera intrinsics and distortion for each camera
        self.camera_params = {}
        for camera_name in self.videos.keys():
            try:
                K, R, t, distCoeffs = self.calib.get_camera_params(camera_name)
                self.camera_params[camera_name] = (K, R, t, distCoeffs)
            except ValueError:
                print(f"Warning: {camera_name} not found in calibration data.")
                self.camera_params[camera_name] = None


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
        if not ret:
            print(f"Frame {frame_number} not found in camera {camera_name}.")
            return None
        
        camera_data = self.camera_params.get(camera_name, None)
        if camera_data is None:
            return frame
        
        K, R, t, distCoeffs = camera_data
        if K is None or R is None or t is None or distCoeffs is None:
            return frame

        h, w = frame.shape[:2]

        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
        frame = cv2.undistort(frame, K, distCoeffs, None, new_K)
        return frame


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