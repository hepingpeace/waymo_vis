import os
import math
import numpy as np
import itertools
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FILENAME = '/media/kaai/One Touch/waymo_vis-1/waymo-od/tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

for data in dataset:
	frame = open_dataset.Frame()
	frame.ParseFromString(bytearray(data.numpy()))
	# print(frame.laser_labels)
'''
camera_labels: 5台摄像机检测到的对象的图像坐标，大小，类型等，从每帧0到4
context: 相机和激光雷达的内部和外部参数，光束倾斜度值
images: 图片
laser_labels: 激光雷达坐标系上物体的XYZ坐标，大小，行进方向，对象类型等等
lasers: 激光点
no_label_zones:非标记区域的设置（有关详细信息，请参阅文档）
pose: 车辆姿势和位置
projected_lidar_labels: 投影由LIDAR检测到的对象时的图像坐标
timestamp_micros: 时间戳
'''

(range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)#解析数据帧
# (point,cp_point) = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)#获取激光点云

# (range_images, camera_projections,
#  _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
#     frame)
print(frame.context)
    
def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in frame.camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

plt.figure(figsize=(25, 20))

for index, image in enumerate(frame.images):
  show_camera_image(image, frame.camera_labels, [3, 3, index+1])