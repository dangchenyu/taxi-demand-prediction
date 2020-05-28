'''
Function: the main precessor of CES cabin demo
Date: 20191016
Author: Yizhang Xia, Chenyu Dang, Chao Zhang
Mail: xiayizhang@minieye.cc
Note:
    must thresh_IoU_detection_between_frames > thresh_IoU_between_BBX
'''

import os
import cv2
import time
import math
import numpy as np
import tensorrt as trt
from xavier_demo_utils import *

# ------------------- parameters -------------------
# global parameters
flag_time_static = True
flag_show_result = True
flag_cameral = False
flag_video = False

#
ind_capture_device = 1
size_cap = [1280, 720]  # width height
# cap = cv2.VideoCapture(ind_capture_device)
cap = cv2.VideoCapture(0)  #rec_20191119_165440.avi
#cap = cv2.VideoCapture('NJ_Y_1_70_20190813_081300_bd02b3da.avi')
#cap = cv2.VideoCapture('NJ_Y_1_70_20190814_081400_28f84997.avi')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_cap[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_cap[1])
# cap.set(cv2.CAP_PROP_FPS, 25)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 1)  # Brightness of the image (only for cameras).
# cap.set(cv2.CAP_PROP_CONTRAST, 1)  # Contrast of the image (only for cameras).
# cap.set(cv2.CAP_PROP_SATURATION, 1)  # Saturation of the image (only for cameras).
# cap.set(cv2.CAP_PROP_HUE, 1)  # Hue of the image (only for cameras).
# cap.set(cv2.CAP_PROP_GAIN, 1)  # Gain of the image (only for cameras).
# cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)  # Exposure (only for cameras).
# cap.set(cv2.CAP_PROP_CONVERT_RGB, False)  # Boolean flags indicating whether images should be converted to RGB.
time_program_total = 0
time_total_detection = 0
time_total_action = 0
time_total_pose = 0
ind_frame = 0
start_ind_frame_want_to_test = 0

# detection parameters
# OUTPUT_NAME = ["conv_45", "conv_53", "conv_61", "conv_69"]
model_info_detection = model_info(deploy_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.prototxt',
                                  model_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.caffemodel',
                                  engine_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.engine',
                                  input_shape=(3,512, 512),
                                  output_name=['conv_blob53', 'conv_blob55','conv_blob57'],
                                  data_type=trt.float32,
                                  flag_fp16=True,
                                  max_workspace_size=1,
                                  max_batch_size=1)

engine_detection = get_engine(model_info_detection)
h_input_detection, d_input_detection, h_output_detection, d_output_detection, stream_detection = allocate_buffers(model_info_detection, engine_detection)
context_detection = engine_detection.create_execution_context()
while True:
    print(ind_frame)
    ind_frame += 1
    ret, frame = cap.read()
    if not ret:
        break
    preprocess_model(frame=frame, pagelocked_buffer=h_input_detection, model_info=model_info_detection)
    do_inference(context_detection, h_input_detection, d_input_detection, h_output_detection, d_output_detection,
                 stream_detection)
    output_box_detection = posprocess_detection(h_output_detection, frame, model_info_detection)






