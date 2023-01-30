import pathlib
import os
import tensorflow as tf
import numpy as np
from matplotlib import patches
import json
import cv2

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_open_dataset import label_pb2
from waymo_open_dataset.camera.ops import py_camera_model_ops
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos import submission_pb2
from waymo_open_dataset.utils import box_utils

OPENLANE_DATA_PATH = '/media/kaai/My Passport/Openlane_data'
WAYMO_PATH = '/media/kaai/Elements/waymo_open_dataset_v_1_3_0'
TRAIN_PATH = '/training'
VALID_PATH = '/validation'
TEST_PATH = '/testing'
# json dir : /media/kaai/Elements/Openlane_data/lane3d_300/training/---/---.json
# image dir : /media/kaai/Elements/Openlane_data/images/training/---/---.jpg
# lidar dir : /media/kaai/Elements/Openlane_data/lidar/training/---/---.npy
# training/asdf/asdf.json, .jpg, .npy


### save paths of data which have at least one 'lane_lines' and one 'vehicles'
### it assume that you have 'lane3d_1000_with_bbox'
for path in [TRAIN_PATH, VALID_PATH]:
    lane_bbox_paths = pathlib.Path('lane3d_1000_with_bbox' + path).rglob('*.json')
    with open(f'data_with_lane_n_bbox_dirs_{path[1:]}.txt', 'w', encoding='utf-8') as txt_f:
        
        for lane_bbox_path in lane_bbox_paths:
            with open(lane_bbox_path, 'r') as lane_bbox_f:
                lane_bbox_json_data = json.load(lane_bbox_f)
                if len(lane_bbox_json_data['lane_lines']) > 0 and len(lane_bbox_json_data['vehicles']) > 0:
                    txt_f.write(str(lane_bbox_path) + '\n')


### make data files 'lane3d_1000_with_bbox'
### it assume that you have 'bbox'
'''for path in [TRAIN_PATH, VALID_PATH]:
    file_lane_paths = pathlib.Path('lane3d_1000' + path).rglob('*.json')
    for lane_path in file_lane_paths:
        folder_name = lane_path.parent.name
        file_name = lane_path.name # 1234.json
        bbox_path = pathlib.Path('bbox' + path) / folder_name / file_name
        lane_bbox_path = pathlib.Path('lane3d_1000_with_bbox' + path) / folder_name / file_name
        with open(lane_path, 'r') as lane_f, open(bbox_path, 'r') as bbox_f:
            lane_json_data = json.load(lane_f)
            bbox_json_data = json.load(bbox_f)
            lane_with_bbox = lane_json_data
            lane_with_bbox['vehicles'] = bbox_json_data['vehicles']
        with open(lane_bbox_path, 'w') as lane_f:
            json.dump(lane_with_bbox, lane_f)'''


### it test whether 'bbox' have correct data
### it assume that you have 'bbox'
'''for path in [TRAIN_PATH, VALID_PATH]:
    file_bbox_paths = pathlib.Path(OPENLANE_DATA_PATH + '/bbox' + path).rglob('*.json')
    for bbox_path in file_bbox_paths:
        folder_name = bbox_path.parent.name
        file_stem = bbox_path.stem
        with open(bbox_path, 'r') as f:
            bbox_json_data = json.load(f)['vehicles']
            for projected_3dbbox in bbox_json_data:
                uv = projected_3dbbox['2d_bbox']
                u, v = uv[:, 0], uv[:, 1]
                img = cv2.imread(bbox_path.parents[3] / 'images' / path[1:] / folder_name / file_stem)
                cv2.imshow('asdf', img)
                cv2.waitKey(0)
                quit()
                '''


### it(including function) makes 'points'(N, 5) data
### it assume that you have Waymo_Open_Dataset
'''def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
        vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
        calibration: Camera calibration details (including intrinsics/extrinsics).
        points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
        Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
        [4, 4])
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant([
        calibration.width,
        calibration.height,
        open_dataset.CameraCalibration.GLOBAL_SHUTTER,], dtype=tf.int32)
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                                camera_image_metadata,
                                                world_points).numpy()

for path in [TRAIN_PATH, VALID_PATH]:
    file_dirs = list(pathlib.Path(WAYMO_PATH + path).glob('*.tfrecord'))
    print('file_length = ', len(file_dirs))
    for file_dir in file_dirs:
        file_dir = str(file_dir)
        file_name = file_dir[len(WAYMO_PATH) + len(path):-len('.tfrecord')]
        new_folder_path = OPENLANE_DATA_PATH + path + '/' + file_name
        pathlib.Path(new_folder_path).mkdir(exist_ok=True)
        # /media/titor/Elements/Openlane_data/lidar + /training + asdf
        tfrecord = tf.compat.v1.data.TFRecordDataset(file_dir, compression_type='')
        for data in tfrecord:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            timestamp_micro = frame.timestamp_micros

            (range_images, camera_projections,
            _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                                                        frame,
                                                        range_images,
                                                        camera_projections,
                                                        range_image_top_pose)
            
            mask = cp_points[0][:, 0] == 1
            points = points[0][mask]
            cp_points = cp_points[0][mask]
            
            camera_ex = np.array(frame.context.camera_calibrations[0].extrinsic.transform)[[3, 7, 11]]
            points[:] -= camera_ex
            cp_points_all_concat = np.concatenate([cp_points[:, 1:3], points], axis=-1)
            
            new_dir = new_folder_path + '/' + str(timestamp_micro) + '00'
            np.save(new_dir + '.npy', cp_points_all_concat) # (..., 5) u(width), v(height), x, y, z
            if path != TEST_PATH:
                assert str(timestamp_micro) + '00.jpg' in os.listdir('/media/kaai/Elements/Openlane_data/images' + path + file_name)
    print(path + ' done')
'''


### it makes 'bbox' of vehicles, pedestrians
### it assume that you have Waymo_Open_Dataset
'''for path in [TRAIN_PATH, VALID_PATH]:
    file_dirs = list(pathlib.Path(WAYMO_PATH + path).glob('*.tfrecord'))
    print('file_length = ', len(file_dirs))
    for file_dir in file_dirs:
        file_dir = str(file_dir)
        file_name = file_dir[len(WAYMO_PATH) + len(path):-len('.tfrecord')]
        new_folder_path = OPENLANE_DATA_PATH + path + '/' + file_name
        pathlib.Path(new_folder_path).mkdir(exist_ok=True)
        # /media/titor/Elements/Openlane_data/lidar + /training + asdf
        tfrecord = tf.compat.v1.data.TFRecordDataset(file_dir, compression_type='')
        for data in tfrecord:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            timestamp_micro = frame.timestamp_micros
            new_dir = new_folder_path + '/' + str(timestamp_micro) + '00'
            
            with open(new_dir + '.json', 'w', encoding='utf-8') as f:
                box_info = {'vehicles':[], 'pedestrian':[]}
                """Displays camera_synced_box 3D labels projected onto camera."""
                # Fetch matching camera calibration.
                calibration = frame.context.camera_calibrations[0]
                for label in frame.laser_labels:
                    box = label.box
                    if label.type != 1 and label.type != 3: # not vehicle and not pedestrian
                        continue
                    FILTER_AVAILABLE = bool(label.num_top_lidar_points_in_box > 0)

                    if not box.ByteSize():
                        # Filter out labels that do not have a camera_synced_box.
                        # waymo dataset version should be more >= 1_3_0
                        continue  
                    if (FILTER_AVAILABLE and not label.num_top_lidar_points_in_box) or (
                        not FILTER_AVAILABLE and not label.num_lidar_points_in_box):
                        continue  # Filter out likely occluded objects.

                    # Retrieve upright 3D box corners.
                    box_coords = np.array([[
                        box.center_x, box.center_y, box.center_z, box.length, box.width,
                        box.height, box.heading
                    ]])
                    corners = box_utils.get_upright_3d_box_corners(
                        box_coords)[0].numpy()  # [8, 3]

                    # Project box corners from vehicle coordinates onto the image.
                    projected_corners = project_vehicle_to_image(frame.pose, calibration,
                                                                corners)
                    u, v, ok = projected_corners.transpose()
                    if label.type == 1:
                        box_info['vehicles'].append({'3d_bbox': corners.tolist(), '2d_bbox': np.column_stack([u, v]).tolist()})
                    else:
                        box_info['pedestrian'].append({'3d_bbox': corners.tolist(), '2d_bbox': np.column_stack([u, v]).tolist()})

                json.dump(box_info, f)'''
                
