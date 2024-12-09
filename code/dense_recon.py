"""Perform dense object reconstruction using SAM2 via space carving"""

import trimesh
import numpy as np
import torch
import sys

sys.path.append('/data/swayam/pose_tracking/sam2')
from sam2.sam2_image_predictor import SAM2ImagePredictor


class DenseObjectReconstructor:
    def __init__(self, voxel_resolution=64):
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

        return voxel_centers


    def project_points(self, points_3d, K, R, t):
        """Project 3D points using pinhole camera model"""
        points_homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

        RT = np.hstack((R, t.reshape(3,1)))

        proj_points = K @ RT @ points_homog.T

        proj_points = proj_points / proj_points[2]
        return proj_points[:2].T


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
        
        for camera_name in video_processor.videos.keys():
            camera_params = next((c for c in calib_data['cameras'] if c['name'] == camera_name), None)
            
            if camera_params is None:
                continue
                
            frame = video_processor.get_frame(camera_name, frame_number)
            if frame is None:
                continue

            K = np.array(camera_params['K'])
            R = np.array(camera_params['R'])
            t = np.array(camera_params['t'])

            proj_points = self.project_points(voxel_centers, K, R, t)

            if camera_name not in detections or not detections[camera_name]:
                continue

            person_detections = [det for det in detections[camera_name] if det['class_id'] == 0]
            if not person_detections:
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

        mesh = trimesh.voxel.ops.points_to_marching_cubes(voxel_centers[voxel_occupancy], 
                                                          pitch=self.voxel_size
                                                        )

        return voxel_centers[voxel_occupancy], voxel_occupancy, mesh