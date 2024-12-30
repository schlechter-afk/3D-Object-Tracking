import json
import numpy as np


class CameraCalibration:
    def __init__(self, calib_file):
        self.calib_file = calib_file
        self.cameras = self._load_calibration_data()


    def _load_calibration_data(self):
        with open(self.calib_file, 'r') as file:
            data = json.load(file)
        return {cam['name']: cam for cam in data['cameras'] if cam['type'] == 'hd'}


    def get_camera_params(self, camera_name):
        camera = self.cameras.get(camera_name)
        if not camera:
            raise ValueError(f"Camera {camera_name} not found in calibration data.")
        K = np.array(camera['K'])
        R = np.array(camera['R'])
        t = np.array(camera['t']).flatten()
        distCoeffs = np.array(camera['distCoef'])
        return K, R, t, distCoeffs