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
model_info_detection = model_info(deploy_file='caffe_models/DLASeg.prototxt',
                                  model_file='caffe_models/DLASeg.caffemodel',
                                  engine_file='caffe_models/DLASeg.engine',
                                  input_shape=(3, 512, 512),
                                  output_name=['conv_blob53', 'conv_blob55','conv_blob57'],
                                  data_type=trt.float32,
                                  flag_fp16=True,
                                  max_workspace_size=1,
                                  max_batch_size=1)

engine_detection = get_engine(model_info_detection)
h_input_detection, d_input_detection, h_output_detection, d_output_detection, stream_detection = allocate_buffers(model_info_detection, engine_detection)
context_detection = engine_detection.create_execution_context()


# -------------------
while True:
    print(ind_frame)
    ind_frame += 1
    if flag_time_static:
        time_program_start = time.time()
   # Capture frame-by-frame
   #  ret, frame = cap.read()
   #  if not ret:
   #      break
    ret, frame = cap.read()
    if not ret:
        break
    if ind_frame < start_ind_frame_want_to_test:
        continue
    # time_start = time.time()
    # print('%.2f' % ((time.time()-time_start)*1000))

    frame_resize = cv2.resize(frame, (size_cap[0], size_cap[1]))
    frame_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    # frame_resize = cv2.resize(frame_gray, (size_cap[0], size_cap[1]))
    if ind_frame % 5 == 0:
        # ------------------- detection block -------------------
        if flag_time_static:
            time_start_detection = time.time()
        preprocess_model(frame_gray=frame_resize, pagelocked_buffer=h_input_detection, model_info=model_info_detection, result_info=result_info_CES)
        do_inference(context_detection, h_input_detection, d_input_detection, h_output_detection, d_output_detection, stream_detection)
        output_box_detection = posprocess_detection(h_output_detection, frame_resize, model_info_detection)
        # connect bbx between frames
        result_info_CES.match_obj_detection(output_box_detection)
        result_info_CES.scan_objs_detection()
        result_info_CES.deal_with_unusual_result_detection(model_info=model_info_detection, test_img=frame_resize)
        if flag_time_static:
            time_total_detection += time.time() - time_start_detection
        
    # -------------------

# ------------------- result parse block -------------------
    result_info_CES.objs_loss_check()
# -------------------

# ------------------- show result block -------------------
    # draw result on image
    # draw_result_on_image(frame_resize, output_box_detection)
    # Display the resulting frame
    if flag_show_result:
        draw_result_on_image(frame_resize, result_info_CES)
        time_start = time.time()
        cv2.imshow('frame', cv2.resize(frame_resize, (640, 320)))
        print('imshow: %.2f' % ((time.time()-time_start)*1000))
        time_start = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('waitKey: %.2f' % ((time.time()-time_start)*1000))


# -------------------

    if flag_time_static:
        time_program_total += time.time() - time_program_start
        print('time_program_total = %.2fms\n'
              'time_total_detection=%.2fms\n'
              'time_total_action=%.2fms\n'
              'time_total_pose=%.2fms\n' %
              (time_program_total/ind_frame*1000,
               time_total_detection/ind_frame*1000,
               time_total_action/ind_frame*1000,
               time_total_pose/ind_frame*1000))

# ------------------- quit program block -------------------
cap.release()
if flag_show_result:
    cv2.destroyAllWindows()
# -------------------

