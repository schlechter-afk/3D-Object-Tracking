"""Perform dense object reconstruction using SAM2 via space carving"""

import trimesh
import numpy as np
import torch
import sys
import cv2
from tqdm import tqdm
sys.path.append('/data/swayam/pose_tracking/sam2')
from sam2.sam2_image_predictor import SAM2ImagePredictor


class DenseObjectReconstructor:
    def __init__(self, voxel_resolution=256):
        """
        Initialize the dense object reconstructor
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            voxel_resolution: Number of voxels along each dimension
        """
        self.voxel_resolution = voxel_resolution
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
        self.min_coords = None
        self.max_coords = None


    def create_voxel_grid(self, bbox_vertices):
        """Create a regular voxel grid within the 3D bounding box"""
        self.min_coords = np.min(bbox_vertices, axis=0)
        self.max_coords = np.max(bbox_vertices, axis=0)
        
        x = np.linspace(self.min_coords[0], self.max_coords[0], self.voxel_resolution)
        y = np.linspace(self.min_coords[1], self.max_coords[1], self.voxel_resolution)
        z = np.linspace(self.min_coords[2], self.max_coords[2], self.voxel_resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z)
        voxel_centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        self.voxel_size = (
            (self.max_coords[0] - self.min_coords[0]) / self.voxel_resolution,
            (self.max_coords[1] - self.min_coords[1]) / self.voxel_resolution,
            (self.max_coords[2] - self.min_coords[2]) / self.voxel_resolution
        )

        return voxel_centers


    # def project_points(self, points_3d, K, R, t):
    #     """
    #     Project 3D points using pinhole camera model
    #     TODO: Check cv2.projectPoints for better performance
    #     """
    #     points_homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    #     RT = np.hstack((R, t.reshape(3,1)))
    #     # TODO: Check w2c vs c2w -> inverse of RT matrix after homogeneous transformation.
    #     # Update: RT is from w2c, so no need to invert it.
    #     proj_points = K @ RT @ points_homog.T

    #     proj_points = proj_points / proj_points[2]
    #     return proj_points[:2].T

    def project_points(self, points_3d, K, R, t, distCoeffs):
        """
        Use OpenCV's projectPoints to handle distortion directly.
        """
        rvec, _ = cv2.Rodrigues(R)
        
        # points_3d = points_3d.reshape((-1, 1, 3)).astype(np.float32)
        
        projected_points, _ = cv2.projectPoints(
            points_3d, rvec, t, K, distCoeffs
        )
        return projected_points.reshape(-1, 2)


    def get_sam_mask(self, image, bbox):
        """Get segmentation mask from SAM"""
        bbox = np.array(bbox)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                box = bbox[None, :],
                multimask_output=True
            )

            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0]
            mask = mask > 0.5

        return mask


    def carve_space(self, voxel_centers, video_processor, calib_data, frame_number, detections):
        """Perform space carving using all available views"""
        voxel_occupancy = np.ones(len(voxel_centers), dtype=bool)
        tqdm_bar = tqdm(video_processor.videos.keys(), desc="Carving space")

        print(f"Lenght of voxel_centers: {len(voxel_centers)}")

        for camera_name in tqdm_bar:
            camera_params = next((c for c in calib_data['cameras'] if c['name'] == camera_name), None)
            
            if camera_params is None:
                print(f"Warning: {camera_name} not found in calibration data.")
                continue
                
            frame = video_processor.get_frame(camera_name, frame_number)
            if frame is None:
                print(f"Frame {frame_number} not found in camera {camera_name}.")
                continue

            K = np.array(camera_params['K'])
            R = np.array(camera_params['R'])
            t = np.array(camera_params['t'])
            distCoeffs = np.array(camera_params['distCoef'])

            proj_points = self.project_points(voxel_centers, K, R, t, distCoeffs)

            if camera_name not in detections or not detections[camera_name]:
                print(f"No detections found for camera {camera_name}.")
                continue

            person_detections = [det for det in detections[camera_name] if det['class_id'] == 0]
            if not person_detections:
                print(f"No person detections found for camera {camera_name}.")
                continue

            combined_mask = np.zeros(frame.shape[:2], dtype=bool)

            for det in person_detections:
                bbox = det['bbox']
                mask = self.get_sam_mask(frame, bbox)
                combined_mask |= mask

            valid_points = (proj_points[:, 0] >= 0) & (proj_points[:, 0] < frame.shape[1]) & \
                         (proj_points[:, 1] >= 0) & (proj_points[:, 1] < frame.shape[0])

            points_valid = proj_points[valid_points].astype(int)
            voxel_occupancy[valid_points] &= combined_mask[points_valid[:, 1], points_valid[:, 0]]

            mask_occup = combined_mask[points_valid[:, 1], points_valid[:, 0]]
            print(f"Mask Occupancy: {np.sum(mask_occup)}")
            print(f"Occupancy: {np.sum(voxel_occupancy)}")

        print(voxel_centers)

        print(voxel_centers[voxel_occupancy])
        mesh = trimesh.voxel.ops.points_to_marching_cubes(voxel_centers[voxel_occupancy], 
                                                          pitch=self.voxel_size
                                                        )

        return voxel_centers[voxel_occupancy], voxel_occupancy, mesh