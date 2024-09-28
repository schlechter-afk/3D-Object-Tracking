import trimesh

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from calibration_data import CameraCalibration
from video_processing import VideoProcessor, DetectionProcessor
# from multiproc_video_proc import VideoProcessor, DetectionProcessor
from object_detection import YOLODetector

DATASET_DIR = '../150821_dance3/'

calib_file = '../150821_dance3/calibration_150821_dance3.json'

video_dir = os.path.join(DATASET_DIR, 'hdVideos')

yolo_model_path = 'yolov5nu.pt'

calib = CameraCalibration(calib_file)
video_processor = VideoProcessor(video_dir)
yolo_detector = YOLODetector(yolo_model_path)

print(f"GPU available: {torch.cuda.is_available()}")

print(f"Device of yolo_detector: {yolo_detector.device}")

detection_processor = DetectionProcessor(video_processor, yolo_detector)

with open(calib_file, 'r') as f:
    calib_data = json.load(f)

class FrustumProjection:
    def __init__(self, K, R, t):
        """
        Initializes the FrustumProjection class with camera intrinsic and extrinsic parameters.
        K: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        """
        self.K = np.array(K)  # (3, 3)
        self.R = np.array(R)  # (3, 3)
        self.t = np.array(t)  # (3, 1)
        self.inv_K = np.linalg.inv(self.K) 
    
    def backproject(self, x, y, depth, K, R, t):
        """
        Backprojects a 2D point into 3D space.
        x, y: 2D point coordinates
        depth: Depth value
        K: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        Returns the 3D point in world coordinates.
        """
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X_cam = (x - cx) * depth / fx
        Y_cam = (y - cy) * depth / fy
        Z_cam = depth
        P_cam = np.array([X_cam, Y_cam, Z_cam])

        # Shape of R: (3, 3), shape of t: (3, 1), shape of P_cam: (3,)
        translation = P_cam - t.flatten()
        P_world = np.dot(R.T, translation)
        # print(f"P_world shape: {P_world.shape}")

        return P_world

    def bbox_to_frustum(self, bbox, near=0.05, far=2000.0):
        """
        Projects the 2D bounding box into a 3D frustum by starting rays from each bbox corner.
        bbox: Bounding box in format [x1, y1, x2, y2]
        near: Near plane distance
        far: Far plane distance
        Returns the vertices of the frustum in world coordinates.
        """
        x1, y1, x2, y2 = bbox  
        
        corner_1_zmin = self.backproject(x1, y1, near, self.K, self.R, self.t)
        corner_2_zmin = self.backproject(x2, y1, near, self.K, self.R, self.t)
        corner_3_zmin = self.backproject(x2, y2, near, self.K, self.R, self.t)
        corner_4_zmin = self.backproject(x1, y2, near, self.K, self.R, self.t)

        corner_1_zmax = self.backproject(x1, y1, far, self.K, self.R, self.t)
        corner_2_zmax = self.backproject(x2, y1, far, self.K, self.R, self.t)
        corner_3_zmax = self.backproject(x2, y2, far, self.K, self.R, self.t)
        corner_4_zmax = self.backproject(x1, y2, far, self.K, self.R, self.t)

        frustum_vertices = np.array([
            corner_1_zmin, corner_2_zmin, corner_3_zmin, corner_4_zmin,
            corner_1_zmax, corner_2_zmax, corner_3_zmax, corner_4_zmax
        ])
        # print(f"Frustum vertices shape: {frustum_vertices.shape}")
        return frustum_vertices

def fix_winding(vertices, faces):
    """
    Ensures that the winding of the faces is consistent.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()  # Ensure normals are correct and consistent
    return mesh.vertices, mesh.faces

class Mesh:
    def __init__(self):
        self.meshes = []
        self.vertices = []
        self.faces = []
        self.meshpaths = []

    def add_frustum(self, frustum_vertices, meshpath):
        """
        Adds a frustum's vertices to the mesh.
        frustum_vertices: Vertices of the frustum (8, 3)
        meshpath: Path to the mesh
        """

        faces = self.compute_frustum_faces(frustum_vertices)
        vertices = frustum_vertices
        vertices, faces = fix_winding(frustum_vertices, faces)
        self.meshes.append(vertices)
        self.vertices.extend(vertices)
        self.faces.extend(faces)
        self.meshpaths.append(meshpath)

    def intersect_meshes(self):
        """
        Computes the intersection of all meshes added so far.
        Returns the intersection points if more than one mesh exists.
        """
        if len(self.meshes) < 2:
            return None

        # load the mesh from meshpath -> initialize a trimesh object
        intersection_mesh = trimesh.load(self.meshpaths[0])

        # append all the meshes from the meshpaths to a list of meshes
        mehes_list = []
        mehes_list.append(intersection_mesh)
        for meshpath in self.meshpaths[1:]:
            mehes_list.append(trimesh.load(meshpath))

        # compute the intersection of all the meshes using trimesh.boolean.intersection
        intersection = trimesh.boolean.intersection(mehes_list, engine='manifold')

        return intersection

    def compute_frustum_faces(self, frustum_vertices):
        """
        Constructs the 12 triangles forming the faces of the frustum.
        frustum_vertices: Vertices of the frustum (8, 3)
        Returns a list of triangles representing the frustum faces.
        """
        faces = [
            [0, 1, 2], [0, 2, 3],  # Near plane
            [4, 5, 6], [4, 6, 7],  # Far plane
            [0, 1, 5], [0, 5, 4],  # Side 1
            [1, 2, 6], [1, 6, 5],  # Side 2
            [2, 3, 7], [2, 7, 6],  # Side 3
            [3, 0, 4], [3, 4, 7]   # Side 4
        ]

        return faces

    def visualize_mesh(self, vertices, faces, title="Mesh Visualization"):
        """
        Visualizes the mesh vertices in a 3D plot.
        vertices: Vertices of the mesh (N, 3)
        faces: Faces of the mesh (list of triangles)
        title: Title of the plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        for face in faces:
            triangle = np.array([vertices[face[0]], vertices[face[1]], vertices[face[2]], vertices[face[0]]])
            ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color="b")
        plt.title(title)
        plt.show()

def save_mesh_to_file(vertices, faces, filename):
    if len(vertices) == 0 or len(faces) == 0:
        print(f"Cannot save mesh to {filename}: Mesh is empty.")
        return
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)


total_frames = 0
for camera in video_processor.videos.keys():
    total_frames = max(total_frames, video_processor.videos[camera].get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > video_processor.videos[camera].get(cv2.CAP_PROP_FRAME_COUNT):
        print(f"Camera {camera} has {video_processor.videos[camera].get(cv2.CAP_PROP_FRAME_COUNT)} frames")

print(f"Total frames: {total_frames}")
total_frames = int(total_frames)

def is_watertight(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.is_watertight

def check_incomplete_faces(faces):
    for face in faces:
        if len(face) != 3:
            return False
    return True

def validate_mesh(vertices, faces):
    if not check_incomplete_faces(faces):
        print("Mesh has incomplete faces.")
        return False
    
    if not is_watertight(vertices, faces):
        print("Mesh is not watertight.")
        return False

    # print("Mesh is valid.")
    return True

import pprint
pprint = pprint.PrettyPrinter(indent=4).pprint
# Processing Loop
for frame in range(total_frames):

    if frame != 1000 and frame != 1001 and frame != 1002:
        continue
    
    import time
    start = time.time()
    detections = detection_processor.process_current_frame_from_all_cameras(frame_number=frame)
    end = time.time()
    pprint(f"BBox detection took {end - start} seconds")

    if detections is None:
        print("Detections is empty")
        continue
    
    start = time.time()

    mesh = Mesh()
    
    for camera in detections.keys():
        camera_params = next((c for c in calib_data['cameras'] if c['name'] == camera)
                            , None)
        
        if camera_params is None:
            print(f"Camera {camera} not found in calibration data")
            continue

        K = camera_params['K']
        R = camera_params['R']
        t = camera_params['t']

        frustum_projection = FrustumProjection(K, R, t)

        for det in detections[camera]:
            if not det:
                continue
            if det['class_id'] != 0:
                continue
            
            bbox = det['bbox']
            frustum_vertices = frustum_projection.bbox_to_frustum(bbox)
            print(f"Frame: {frame}")
            frame_dir = os.makedirs(f"output/{frame}", exist_ok=True)
            current_mesh_path = f"{frame_dir}/frustum_camera_{camera}_frame_{frame}.ply"
            print(f"CMesh path: {current_mesh_path}")
            mesh.add_frustum(frustum_vertices, current_mesh_path)

            current_mesh = Mesh()

            current_mesh.add_frustum(frustum_vertices, current_mesh_path)
            trimesh_current_mesh = trimesh.Trimesh(vertices=current_mesh.vertices, faces=current_mesh.faces)

            if validate_mesh(current_mesh.vertices, current_mesh.faces):
                # print("Mesh is valid and ready for further processing.")
                pass
            else:
                print("Mesh validation failed.")
                break
            
            if not trimesh_current_mesh.is_winding_consistent:
                print("Mesh has non-manifold edges or vertices.")
                break

            save_mesh_to_file(current_mesh.vertices, current_mesh.faces, current_mesh_path)

    pprint(f"Computing all meshes for frame: {frame} took: {time.time() - start} seconds")
    start = time.time()
    intersection = mesh.intersect_meshes()
    end = time.time()
    pprint(f"Intersection took {end - start} seconds")
    
    if intersection is not None:
        save_mesh_to_file(intersection.vertices, intersection.faces, f"intersection_frame_{frame}.ply")
    else:
        print("Intersection could not be computed.")

    # Load the intersection mesh's vertices
    intersection_mesh_vertices = intersection.vertices

    xmincoord = np.min(intersection_mesh_vertices[:, 0])
    ymincoord = np.min(intersection_mesh_vertices[:, 1])
    zmincoord = np.min(intersection_mesh_vertices[:, 2])

    xmaxcoord = np.max(intersection_mesh_vertices[:, 0])
    ymaxcoord = np.max(intersection_mesh_vertices[:, 1])
    zmaxcoord = np.max(intersection_mesh_vertices[:, 2])

    threeDimensionalBBox = np.array([(xmincoord, ymincoord, zmincoord), (xmincoord, ymincoord, zmaxcoord), 
                        (xmincoord, ymaxcoord, zmincoord), (xmincoord, ymaxcoord, zmaxcoord),
                        (xmaxcoord, ymincoord, zmincoord), (xmaxcoord, ymincoord, zmaxcoord), 
                        (xmaxcoord, ymaxcoord, zmincoord), (xmaxcoord, ymaxcoord, zmaxcoord)])

    threeDimensionalBBoxMesh = trimesh.Trimesh(vertices=threeDimensionalBBox, faces=[[0, 1, 2], [1, 2, 3], [2, 3, 7], [2, 6, 7], [0, 2, 6], [0, 4, 6], [0, 1, 5], [0, 4, 5], [4, 5, 7], [5, 6, 7], [1, 5, 7], [1, 3, 7]])
    threeDimensionalBBoxMesh.export(f"3D_BBox_frame_{frame}.ply")

    # break