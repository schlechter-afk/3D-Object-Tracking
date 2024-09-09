import os
import cv2
from calibration_data import CameraCalibration
from video_processing import VideoProcessor, DetectionProcessor
from object_detection import YOLODetector
from mesh import Mesh, FrustumProjection

DATASET_DIR = '../150821_dance3/'

def visualize_mesh(vertices, title="Mesh Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    plt.title(title)
    plt.show()

def main():
    calib_file = '../150821_dance3/calibration_150821_dance3.json'

    video_dir = os.path.join(DATASET_DIR, 'hdVideos')

    yolo_model_path = 'yolov8n.pt'

    calib = CameraCalibration(calib_file)

    video_processor = VideoProcessor(video_dir)
    yolo_detector = YOLODetector(yolo_model_path)

    detection_processor = DetectionProcessor(video_processor, yolo_detector)

    detections_per_frame = detection_processor.process_all_frames()

    print("All detections processed.\n\n\n\n\n\n")

    cnt = 0

    for frame_number, detections_list in detections_per_frame.items():
        cnt += 1
        if cnt <= 500:
            continue

        mesh = Mesh()
        print(f"Now mesh creation frame {frame_number}...")
        cv2.imwrite(f'frame_{frame_number}.jpg', video_processor.get_frame('00_00', frame_number))
        break

        for detection in detections_list:

            camera_name = detection['camera_name']
            detections = detection['detections']
            
            camera_params = next((c for c in calib_data['cameras'] if c['name'] == camera_name)
                                , None)

            if camera_params:
                frustum_projection = FrustumProjection(
                    camera_params['K'],
                    camera_params['R'],
                    camera_params['t']
                )
                
                for det in detections:
                    bbox = det['bbox']
                    frustum_vertices = frustum_projection.bbox_to_frustum(bbox)
                    mesh.add_frustum(frustum_vertices)

        intersection = mesh.intersect_meshes()
        if intersection is not None:
            visualize_mesh(intersection, title=f"Intersection at Frame {frame_number}")

        break
    
    video_processor.release()

if __name__ == '__main__':
    main()