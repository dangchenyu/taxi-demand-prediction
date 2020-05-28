'''
Function: the function for CES cabin demo
Date: 20191031
Author: Yizhang Xia, Chenyu Dang, Chao Xu
Mail: xiayizhang@minieye.cc
Note:

'''

import cv2
import math
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
import torch.nn as nn
from sklearn import linear_model
import skimage.measure
# do not delete the next import https://devtalk.nvidia.com/default/topic/1038494/tensorrt/logicerror-explicit_context_dependent-failed-invalid-device-context-no-currently-active-context-/
import pycuda.autoinit
import time

# ------------------- static parameters -------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
REG_LOCATION_DETECTION = linear_model.LinearRegression()
FONT_LABEL = cv2.FONT_HERSHEY_SIMPLEX
COLOR_LABEL = (0, 0, 0)
LINE_WIDTH_LABEL = 3
FONT_SCALE = 1.5


# Non-Maximum Suppression for removing redundant bounding box
def calculate_NMS(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# to calculate Intersection-over-Union
def calculate_IoU(candidate_bounding_box, ground_truth_bounding_box):
    # candidate_bounding_box = [x1, y1, x2, y2,...]
    cx1 = float(candidate_bounding_box[0])
    cy1 = float(candidate_bounding_box[1])
    cx2 = float(candidate_bounding_box[2])
    cy2 = float(candidate_bounding_box[3])

    gx1 = float(ground_truth_bounding_box[0])
    gy1 = float(ground_truth_bounding_box[1])
    gx2 = float(ground_truth_bounding_box[2])
    gy2 = float(ground_truth_bounding_box[3])

    carea = (cx2 - cx1) * (cy2 - cy1)
    garea = (gx2 - gx1) * (gy2 - gy1)

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    iou = area / (carea + garea - area)
    return iou


# get engine from loading the existing engine or building a new engine
def get_engine(model_info):
    # To apply for space
    def GiB(val):
        return val * 1 << 30

    # build engine based on caffe
    def build_engine_caffe(model_info):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
            builder.max_batch_size = model_info.max_batch_size
            builder.max_workspace_size = GiB(model_info.max_workspace_size)
            builder.fp16_mode = model_info.flag_fp16

            # Parse the model and build the engine.
            model_tensors = parser.parse(deploy=model_info.deploy_file, model=model_info.model_file, network=network,
                                         dtype=model_info.data_type)
            for ind_out in range(len(model_info.output_name)):
                print(model_info.output_name[ind_out])
                network.mark_output(model_tensors.find(model_info.output_name[ind_out]))
            print("Building TensorRT engine. This may take few minutes.")
            return builder.build_cuda_engine(network)

    try:
        with open(model_info.engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            print('-------------------load engine-------------------')
            return runtime.deserialize_cuda_engine(f.read())
    except:
        # Fallback to building an engine if the engine cannot be loaded for any reason.
        engine = build_engine_caffe(model_info)
        with open(model_info.engine_file, "w+") as f:
            f.write(engine.serialize())
            print('-------------------save engine-------------------')
        return engine


# allocate buffers
def allocate_buffers(model_info, engine):
    h_output = []
    d_output = []
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(model_info.data_type))
    # h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    for ind_out in range(len(model_info.output_name)):
        h_output_temp = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(ind_out + 1)),
                                              dtype=trt.nptype(model_info.data_type))
        h_output.append(h_output_temp)

    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    for ind_out in range(len(model_info.output_name)):
        d_output_temp = cuda.mem_alloc(h_output[ind_out].nbytes)
        d_output.append(d_output_temp)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    return h_input, d_input, h_output, d_output, stream


# model feedforward
def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    bindings = [int(d_input)]
    for ind_out in range(len(d_output)):
        bindings.append(int(d_output[ind_out]))
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.

    for ind_out in range(len(d_output)):
        cuda.memcpy_dtoh_async(h_output[ind_out], d_output[ind_out], stream)
    # Synchronize the stream
    stream.synchronize()


# get detected result to send to other task
def get_input_patch(result_info):
    input_patch = []
    for ind_obj in range(len(result_info.objs)):
        if 1 == result_info.objs[ind_obj].type_obj:
            content_label_predict = str(result_info.objs[ind_obj].ID)
            x1 = result_info.objs[ind_obj].history_location[-1][0]
            y1 = result_info.objs[ind_obj].history_location[-1][1]
            x2 = result_info.objs[ind_obj].history_location[-1][2]
            y2 = result_info.objs[ind_obj].history_location[-1][3]
            input_patch.append([x1, y1, x2, y2, content_label_predict])
    return input_patch


# preprocess of imagecd l
def preprocess_model(frame, pagelocked_buffer, model_info):
    frame_resize = cv2.resize(frame, (model_info.input_shape[2], model_info.input_shape[1]))
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
    inp_image = ((frame_resize / 255. - mean) / std)
    frame_nor = inp_image.transpose([2, 0, 1]).astype(trt.nptype(model_info.data_type)).ravel()
    np.copyto(pagelocked_buffer, frame_nor)


# def preprocess_model_action_expend(frame_gray, pagelocked_buffer, model_info, result_info, ind_objs=-1, crop=False,
#                                    expend_crop=False):
#     temp = frame_gray.copy()
#     print('temp_shape------', temp.shape)
#     if crop and expend_crop:
#         size = [temp.shape[0], temp.shape[1]]
#         x1 = int(result_info.objs[ind_objs].history_location[-1][0])
#         x2 = int(result_info.objs[ind_objs].history_location[-1][2])
#         y1 = int(result_info.objs[ind_objs].history_location[-1][1])
#         y2 = int(result_info.objs[ind_objs].history_location[-1][3])
#         w,h=x2-x1,y2-y1
#         print('crop_res_x_y_1',[x1,y1,x2,y2])
#         #####if at the left expend 50 and
#         if x1 <= size[0] // 2 and x1<=160:
#             x1, y1 = (x1 - 50),y1-20
#             x2, y2 = x1 + w, y1 + h
#             if x1 <= 0:
#                 x1 = 1
#             if y1 <= 0 :
#                 y1 = 1
#         #####if at the light expend 50
#         elif (x1 + w)>= size[0] // 2 and (x2+w)>=(size[0]-(size[1]//4)):
#             x2, y2 = x1 + w + size[0]//4, y1 + h + 20
#             if x2 >= size[1]:
#                 x2 = size[1] - 1
#             if y2 >= size[0]:
#                 y2 = size[0] - 1
#         print('preprocess_model_action_expend',[x1,y1,x2,y2])
#         print(temp.shape)
#         temp= temp[x1:x2, y1:y2]
#     frame_resize = cv2.resize(temp, (model_info.input_shape[2], model_info.input_shape[1]))
#     frame_expand = np.expand_dims(frame_resize, axis=-1)
#     frame_expand = frame_expand / 256.0
#     frame_nor = frame_expand.transpose([2, 0, 1]).astype(trt.nptype(model_info.data_type)).ravel()
#     np.copyto(pagelocked_buffer, frame_nor)
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
def _nms(heat, kernel=3):
    keep=np.zeros_like(heat)
    hmax_person=pool2d(heat[0][0], kernel_size=3, stride=1, padding=1, pool_mode='max')
    keep[0][0] = hmax_person == heat[0][0]
    return heat * keep
# def _nms(heat, kernel=3):
#     pad = (kernel - 1) // 2
#
#     hmax = nn.functional.max_pool2d(
#         torch.from_numpy(heat), (kernel, kernel), stride=1, padding=pad)
#     keep = (hmax.numpy() == heat)
#     return heat * keep
def _transpose_and_gather_feat(feat, ind):
    feat= feat.transpose(0,2,3,1)
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = np.expand_dims(feat[0][ind[0]],axis=0)
    return feat
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def _topk(scores, K=40):
    batch, cat, height, width = scores.shape
    score_reshape=scores.reshape(batch, cat, -1)
    topk_inds_people = np.expand_dims(score_reshape[0][0].argsort()[-K:][::-1],axis=0)
    topk_score_people=score_reshape[0][0][topk_inds_people]
    topk_inds_people = topk_inds_people % (height * width)
    topk_ys = (topk_inds_people / width).astype(np.int)
    topk_xs = (topk_inds_people % width).astype(np.int)


    topk_clses = np.zeros((1,K))


    return topk_score_people, topk_inds_people, topk_clses, topk_ys, topk_xs
def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.shape
    # perform nms on heatmaps
    heat = _nms(heat)


    scores, inds, clses, ys, xs = _topk(heat, K=K)

    reg = _transpose_and_gather_feat(reg, inds)
    xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]

    wh = _transpose_and_gather_feat(wh, inds)

    clses = clses.reshape(batch, K, 1)
    scores = scores.reshape(batch, K, 1)
    bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2],axis=2)
    detections = np.concatenate([bboxes, scores, clses], axis=2)

    return detections


def posprocess_detection(h_output, test_img, model_info):
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    hm = sigmoid(h_output[0].reshape(1, 80, 128, 128))
    wh = h_output[1].reshape(1, 2, 128, 128)
    reg = h_output[2].reshape(1, 2, 128, 128)
    dets = ctdet_decode(hm, wh, reg=reg)

    temp = 0

    # dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    #
    # if return_time:
    #     return output, dets, forward_time
    # else:
    #     return output, dets
    #
    #     return output_box


# pose processaction
def posprocess_action(h_output, model_info):
    out_predict = []
    for ind_out in range(len(h_output)):
        out_predict.append(np.argmax(np.array(h_output[ind_out].reshape((model_info.num_class_action[ind_out])))))
    return out_predict


# poseprocess pose
def posprocess_pose(h_output_pose, model_info_pose):
    heat_map_scale = model_info_pose.input_shape[1] / model_info_pose.multiple_downsample_pose
    batch_heatmaps = h_output_pose[0].reshape((1, model_info_pose.num_points_pose, heat_map_scale, heat_map_scale))

    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    maxvals = np.reshape(maxvals, (10))
    preds *= pred_mask
    return preds, maxvals
    # smooth pose part


def get_pose_ava(joints_queue):
    add_queue = np.array([0])
    for item in joints_queue:
        add_queue = np.add(add_queue, item)
    joints_ava = add_queue / len(joints_queue)
    return joints_ava


def get_pose_new(joints_ava, pred, point_dis):
    distance = np.sqrt(np.square(joints_ava - pred).sum(2))
    wrong_ind = np.where(distance > point_dis)
    pred[wrong_ind] = joints_ava[wrong_ind]
    return pred


def predict_obj_keypoints(joints_queue):
    joints_queue_array = np.array(joints_queue)
    train_x = np.array(range(len(joints_queue))).reshape(-1, 1)
    train_y = joints_queue_array.reshape(-1, 20)
    test_x = np.array([len(joints_queue) + 1]).reshape(-1, 1)
    REG_LOCATION_DETECTION.fit(train_x, train_y)
    predict = REG_LOCATION_DETECTION.predict(test_x)
    predict = predict.reshape(-1, 2)
    predict = np.expand_dims(predict, 0)
    return predict


def max_list(list1, list2):
    if len(list1) >= len(list2):
        return list1
    else:
        return list2


def max_list4(a1, b1, c1, d1):
    listall = [a1, b1, c1, d1]
    list = [len(a1), len(b1), len(c1), len(d1)]
    return listall[list.index(max(list))]


def compare_result_info(list1):
    history_update = []
    adult = []
    kid = []
    a, b, c, d = [], [], [], []
    hand_in, hand_out = [], []
    head_in, head_out = [], []
    for i in range(len(list1)):
        if list1[i][0] == 1:
            kid.append(1)
        else:
            adult.append(0)
        if list1[i][1] == 0:
            hand_in.append(0)
        else:
            hand_out.append(1)
        if list1[i][2] == 0:
            head_in.append(0)
        else:
            head_out.append(1)
        if list1[i][3] == 0:
            a.append(0)
        elif list1[i][3] == 1:
            b.append(1)
        elif list1[i][3] == 2:
            c.append(2)
        else:
            d.append(3)
    history_update.append(max_list(adult, kid)[0])
    history_update.append(max_list(hand_in, hand_out)[0])
    history_update.append(max_list(head_in, head_out)[0])
    history_update.append(max_list4(a, b, c, d)[0])
    return history_update


# to store model information
class model_info(object):
    def __init__(self, deploy_file, model_file, engine_file, input_shape=(3, 512, 512),
                 output_name=["conv_45", "conv_53"], data_type=trt.float32, flag_fp16=True, max_workspace_size=1,
                 max_batch_size=1):
        self.deploy_file = deploy_file
        self.model_file = model_file
        self.engine_file = engine_file
        self.data_type = data_type
        self.flag_fp16 = flag_fp16
        self.output_name = output_name
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.confidence = -1.0
        self.thresh_IoU_between_BBX = -1.0
        self.num_class = -1
        self.anchor_w = []
        self.anchor_h = []
        self.crop_box = []  # [[x1, y1, x2, y2], [x1, y1, x2, y2]]
        self.num_points_pose = None
        self.multiple_downsample_pose = None
        self.num_class_action = None  # [age, hand, head, pose_action]
        self.thresh_drop_small_obj = -1

    def set_anchor(self, anchor):
        for ind_anchor in range(len(anchor) // 2):
            self.anchor_w.append(anchor[ind_anchor * 2])
            self.anchor_h.append(anchor[ind_anchor * 2 + 1])

    def set_num_class(self, num_class):
        self.num_class = num_class

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_thresh_IoU_between_BBX(self, thresh_IoU_between_BBX):
        self.thresh_IoU = thresh_IoU_between_BBX

    def set_thresh_drop_small_obj(self, thresh_drop_small_obj):
        self.thresh_drop_small_obj = thresh_drop_small_obj

    def set_crop_box(self, crop_box):
        self.crop_box = crop_box

    def set_num_points_pose(self, num_points_pose):
        self.num_points_pose = num_points_pose

    def set_point_dis(self, point_dis):
        self.point_dis = point_dis

    def set_multiple_downsample_pose(self, multiple_downsample_pose):
        self.multiple_downsample_pose = multiple_downsample_pose

    def set_num_class_action(self, num_class_action=[2, 2, 2,
                                                     4]):  # [[adult=0, child=1], [hand_in=0, hand_out=1], [head_in=0, head_out=1], [sit=0, squat=1, stand=2, lie=3]]
        self.num_class_action = num_class_action


class result_info(object):
    def __init__(self,
                 ID_max=0,
                 thresh_IoU_detection_between_frames_detecion=0.5,
                 num_frame_is_obj_detection=5,
                 num_frame_delete_disappear_obj_detection=3,
                 num_frame_for_prediction_location_detection=4,
                 thresh_num_history_want_to_delete_detection=64,
                 num_history_want_to_retain_detection=8,
                 thresh_num_objs_loss=50,
                 point_dis=15,
                 smooth_frames=5,
                 hold_frames=5,
                 show_threshold=0.5,
                 predict_type='average'):
        self.objs = []
        self.point_set = np.array([[0, 1], [2, 3], [2, 9],
                                   [3, 4], [5, 6], [8, 7],
                                   [9, 8]])
        self.show_threshold = show_threshold
        self.ID_max = ID_max
        self.thresh_IoU_detection_between_frames_detecion = thresh_IoU_detection_between_frames_detecion
        self.num_frame_is_obj_detection = num_frame_is_obj_detection
        self.num_frame_delete_disappear_obj_detection = num_frame_delete_disappear_obj_detection
        self.num_frame_for_prediction_location_detection = num_frame_for_prediction_location_detection
        self.thresh_num_history_want_to_delete_detection = thresh_num_history_want_to_delete_detection
        self.num_history_want_to_retain_detection = num_history_want_to_retain_detection
        self.num_objs_loss = 0
        self.thresh_num_objs_loss = thresh_num_objs_loss
        self.point_dis = point_dis
        self.smooth_frames = smooth_frames
        self.hold_frames = hold_frames
        self.vote_frame = 7
        self.predict_type = predict_type

    def _add_obj_detection(self, obj_new):
        self.objs.append(obj_new)
        self.ID_max += 1

    def match_obj_detection(self, output_box_detection):
        for ind_output_box_detection in range(len(output_box_detection)):
            IoU_max = -1
            ind_obj_max = -1
            flag_matched = False
            for ind_obj in range(len(self.objs)):
                temp_IoU = calculate_IoU(output_box_detection[ind_output_box_detection],
                                         self.objs[ind_obj].history_location[-1])
                if self.thresh_IoU_detection_between_frames_detecion < temp_IoU:
                    if temp_IoU > IoU_max:
                        IoU_max = temp_IoU
                        ind_obj_max = ind_obj
                        flag_matched = True
            if flag_matched:
                self.objs[ind_obj_max].add_history_location(output_box_detection[ind_output_box_detection][0:4])
            else:
                self._add_obj_detection(object_CES(ID=self.ID_max,
                                                   type_obj=output_box_detection[ind_output_box_detection][-1],
                                                   location_obj=output_box_detection[ind_output_box_detection][0:4],
                                                   num_frame_for_prediction_location_detection=self.num_frame_for_prediction_location_detection,
                                                   thresh_num_history_want_to_delete_detection=self.thresh_num_history_want_to_delete_detection,
                                                   num_history_want_to_retain_detection=self.num_history_want_to_retain_detection))

    def scan_objs_detection(self):
        for _ in range(len(self.objs)):  # the loop for reset loop when delete obj
            for ind_obj in range(len(self.objs)):
                if not self.objs[ind_obj].state_matched_detection:
                    if self.objs[ind_obj].length_history_detection >= self.num_frame_is_obj_detection:
                        if self.objs[ind_obj].num_disappeared >= self.num_frame_delete_disappear_obj_detection:
                            del (self.objs[ind_obj])
                            break  # jump out the inner loop
                        else:
                            self.objs[ind_obj].predict_obj_location_detection(self.objs[ind_obj])
                    else:
                        del (self.objs[ind_obj])
                        self.ID_max -= 1
                        break  # jump out the inner loop

        for ind_objs in range(len(self.objs)):  # the loop for reset loop when delete obj
            self.objs[ind_objs].state_matched_detection = False

    def deal_with_unusual_result_detection(self, model_info, test_img):
        for _ in range(len(self.objs)):
            for ind_obj in range(len(self.objs)):
                if test_img.shape[1] < self.objs[ind_obj].history_location[-1][0]:  # x1
                    self.objs[ind_obj].history_location[-1][0] = test_img.shape[1] - 1
                if test_img.shape[0] < self.objs[ind_obj].history_location[-1][1]:  # y1
                    self.objs[ind_obj].history_location[-1][1] = test_img.shape[0] - 1
                if test_img.shape[1] < self.objs[ind_obj].history_location[-1][2]:  # x2
                    self.objs[ind_obj].history_location[-1][2] = test_img.shape[1] - 1
                if test_img.shape[0] < self.objs[ind_obj].history_location[-1][3]:  # y2
                    self.objs[ind_obj].history_location[-1][3] = test_img.shape[0] - 1
                for ind_lable_location in range(len(self.objs[ind_obj].history_location[-1])):
                    if 0 > self.objs[ind_obj].history_location[-1][ind_lable_location]:  # x1
                        self.objs[ind_obj].history_location[-1][ind_lable_location] = 0
                if ((self.objs[ind_obj].history_location[-1][2] - self.objs[ind_obj].history_location[-1][
                    0]) < model_info.thresh_drop_small_obj) or \
                        ((self.objs[ind_obj].history_location[-1][3] - self.objs[ind_obj].history_location[-1][
                            1]) < model_info.thresh_drop_small_obj):  # drop small obj
                    del (self.objs[ind_obj])
                    self.ID_max -= 1
                    break

    def _convert_abs_pos(self, keypoints, ind_objs, model_info):

        x1 = self.objs[ind_objs].history_location[-1][0]
        y1 = self.objs[ind_objs].history_location[-1][1]
        x2 = self.objs[ind_objs].history_location[-1][2]
        y2 = self.objs[ind_objs].history_location[-1][3]
        w = x2 - x1
        h = y2 - y1
        scale_x = w / model_info.input_shape[1] * model_info.multiple_downsample_pose
        scale_y = h / model_info.input_shape[2] * model_info.multiple_downsample_pose
        new_keypoints = (keypoints * [scale_x, scale_y])
        new_keypoints = np.add(new_keypoints, np.array([x1, y1]))
        return new_keypoints

    def add_pose(self, keypoints, ind_objs, model_info):

        new_keypoints = self._convert_abs_pos(keypoints, ind_objs, model_info)
        if len(self.objs[ind_objs].history_pose) < self.smooth_frames:
            self.objs[ind_objs].history_pose.append(new_keypoints)
        else:
            self.update_queue(new_keypoints, ind_objs, "key_points")

    def add_pose_conf(self, conf, ind_objs):
        if len(self.objs[ind_objs].history_pose_conf) < self.hold_frames:
            self.objs[ind_objs].history_pose_conf.append(conf)
        else:
            self.update_queue(conf, ind_objs, "conf")

    def update_queue(self, obj, ind_objs, task):
        if task == "key_points":
            self.objs[ind_objs].history_pose.pop(0)
            self.objs[ind_objs].history_pose.append(obj)
        else:
            self.objs[ind_objs].history_pose_conf.pop(0)
            self.objs[ind_objs].history_pose_conf.append(obj)

        ######################add_actions

    def add_action(self, objs_action_age, objs_action_hand, objs_action_head, objs_action_pose, ind_objs):
        if len(self.objs[ind_objs].history_pose) < 5:
            self.objs[ind_objs].history_action.append(
                [objs_action_age[-1], objs_action_hand[-1], objs_action_head[-1], objs_action_pose[-1]])
        else:
            self.update_queue_action_mutil(objs_action_age, objs_action_hand, objs_action_head, objs_action_pose,
                                           ind_objs)

    # def add_action(self, objs_action_age, objs_action_hand, objs_action_head, objs_action_pose, ind_objs):
    #     self.objs[ind_objs].history_action=[objs_action_age[-1], objs_action_hand[-1], objs_action_head[-1], objs_action_pose[-1]]

    def update_queue_action_mutil(self, objs_action_age, objs_action_hand, objs_action_head, objs_action_pose,
                                  ind_objs):
        self.objs[ind_objs].history_action.pop(0)
        self.objs[ind_objs].history_action.append(
            [objs_action_age[-1], objs_action_hand[-1], objs_action_head[-1], objs_action_pose[-1]])

    def _accumulate_num_objs_loss(self):
        self.num_objs_loss += 1

    def _reset_num_objs_loss(self):
        self.num_objs_loss = 0

    def objs_loss_check(self):
        flag_have_person_temp = False
        for ind_objs in range(len(self.objs)):
            if 1 == self.objs[ind_objs].type_obj:
                flag_have_person_temp = True
                self._reset_num_objs_loss()
        if not flag_have_person_temp:
            self._accumulate_num_objs_loss()


class object_CES(object):
    def __init__(self, ID, type_obj, location_obj, num_frame_for_prediction_location_detection=3,
                 thresh_num_history_want_to_delete_detection=64, num_history_want_to_retain_detection=8,
                 age_person=-1,
                 num_disappeared=0, length_history_detection=0):
        self.ID = ID
        self.type_obj = type_obj  # 0=person, 1=wallet, 2=phone, 3=handbag
        self.age_person = age_person  # -1=except_person 0=adult, 1=child
        self.history_location = []  # [[x1, y1, x2, y2], [x1, y1, x2, y2]]
        self.history_action = []  # [[], []]
        self.history_pose = []  # [[], []]
        self.history_pose_conf = []  # [[]]
        self.num_disappeared = num_disappeared
        self.length_history_detection = length_history_detection
        self.state_matched_detection = True
        self.num_frame_for_prediction_location_detection = num_frame_for_prediction_location_detection
        self.num_history_want_to_retain_detection = num_history_want_to_retain_detection
        self.thresh_num_history_want_to_delete_detection = thresh_num_history_want_to_delete_detection

        self.add_history_location(location_obj)

    def drop_outdated_history(self):
        del (self.history_location[
             :self.thresh_num_history_want_to_delete_detection - self.num_history_want_to_retain_detection + 1])
        del (self.history_action[
             :self.thresh_num_history_want_to_delete_detection - self.num_history_want_to_retain_detection + 1])
        del (self.history_pose[
             :self.thresh_num_history_want_to_delete_detection - self.num_history_want_to_retain_detection + 1])
        self.length_history_detection = self.num_history_want_to_retain_detection

    def add_history_location(self, location_obj):
        self.history_location.append(location_obj)
        self.add_length_history_detection()
        self.set_state_matched_detection(state_matched_detection=True)

        if self.length_history_detection > self.thresh_num_history_want_to_delete_detection:
            self.drop_outdated_history()

    def add_history_action(self, action_obj):
        self.history_action.append(action_obj)

    def add_num_disappeared(self):
        self.num_disappeared += 1
        self.set_state_matched_detection(state_matched_detection=False)

    def add_length_history_detection(self):
        self.length_history_detection += 1

    def set_state_matched_detection(self, state_matched_detection):
        self.state_matched_detection = state_matched_detection

    def predict_obj_location_detection(self, obj):
        train_x = np.array(range(self.num_frame_for_prediction_location_detection)).reshape(-1, 1)
        train_y = np.array(obj.history_location[-self.num_frame_for_prediction_location_detection:]).reshape(-1, 4)
        test_x = np.array([5]).reshape(-1, 1)
        REG_LOCATION_DETECTION.fit(train_x, train_y)
        self.add_history_location(REG_LOCATION_DETECTION.predict(test_x)[0])
        self.add_num_disappeared()


###draw_warning####
def stick_warning(sour_img=None, warning_im_path=None):
    warning_img = cv2.imread(warning_im_path)
    img = warning_img
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (56, 56))
    w, h = sour_img.shape[0] / 2, sour_img.shape[1] / 2
    sour_img[w - 56:w, h - 56:h] = im
    return sour_img


# draw result on image

def draw_result_on_image(frame_resize, result_info):
    for ind_obj in range(len(result_info.objs)):

        # ------------------- detection block -------------------
        if result_info.objs[ind_obj].length_history_detection > result_info.num_frame_is_obj_detection:
            content_label_predict = str(result_info.objs[ind_obj].ID)
            x1 = result_info.objs[ind_obj].history_location[-1][0]
            y1 = result_info.objs[ind_obj].history_location[-1][1]
            x2 = result_info.objs[ind_obj].history_location[-1][2]
            y2 = result_info.objs[ind_obj].history_location[-1][3]

            cv2.rectangle(frame_resize, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_LABEL, LINE_WIDTH_LABEL)

            location_label_predict = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))

            if 1 == result_info.objs[ind_obj].type_obj:  # person
                content_label_predict = content_label_predict + '_person'
            elif 2 == result_info.objs[ind_obj].type_obj:  # person
                content_label_predict = content_label_predict + '_wallet'
            elif 3 == result_info.objs[ind_obj].type_obj:  # person
                content_label_predict = content_label_predict + '_phone'
            elif 4 == result_info.objs[ind_obj].type_obj:  # person
                content_label_predict = content_label_predict + '_handbag'
            # ------------------- action block -------------------
            ##no_time logic
            '''
            if 1 == result_info.objs[ind_obj].type_obj:
                temp_history_action = result_info.objs[ind_obj].history_action
                print('temp_history_action',temp_history_action)
               # update_frame_info = compare_result_info(temp_history_action)
                # [[adult=0, child=1], [hand_in=0, hand_out=1], [head_in=0, head_out=1], [sit=0, squat=1, stand=2, lie=3]]
                for ind_task_action in range(len(temp_history_action)):
                    if 0 == ind_task_action:
                        if 0 == temp_history_action[ind_task_action]:
                            content_label_predict += '_adult'
                        elif 1 == temp_history_action[ind_task_action]:
                            content_label_predict += '_child'
                    if 1 == ind_task_action:
                        if 0 == temp_history_action[ind_task_action]:
                            content_label_predict += '_hand-in'
                        elif 1 == temp_history_action[ind_task_action]:
                            content_label_predict += '_hand-out'
                    if 2 == ind_task_action:
                        if 0 == temp_history_action[ind_task_action]:
                            content_label_predict += '_head-in'
                        elif 1 == temp_history_action[ind_task_action]:
                            content_label_predict += '_head-out'
                    if 3 == ind_task_action:
                        if 0 == temp_history_action[ind_task_action]:
                            content_label_predict += '_sit'
                        elif 1 == temp_history_action[ind_task_action]:
                            content_label_predict += '_squat'
                        elif 2 == temp_history_action[ind_task_action]:
                            content_label_predict += '_stand'
                        elif 3 ==temp_history_action[ind_task_action]:
                            content_label_predict += '_lie'            
            '''
            if 1 == result_info.objs[ind_obj].type_obj:
                temp_history_action = result_info.objs[ind_obj].history_action
                update_frame_info = compare_result_info(temp_history_action)
                print('temp_history_action', temp_history_action, update_frame_info)
                # [[adult=0, child=1], [hand_in=0, hand_out=1], [head_in=0, head_out=1], [sit=0, squat=1, stand=2, lie=3]]
                for ind_task_action in range(len(update_frame_info)):
                    if 0 == ind_task_action:
                        if 0 == update_frame_info[ind_task_action]:
                            content_label_predict += '_adult'
                        elif 1 == update_frame_info[ind_task_action]:
                            content_label_predict += '_child'
                    if 1 == ind_task_action:
                        if 0 == update_frame_info[ind_task_action]:
                            content_label_predict += '_hand-in'
                        elif 1 == update_frame_info[ind_task_action]:
                            content_label_predict += '_hand-out'
                    if 2 == ind_task_action:
                        if 0 == update_frame_info[ind_task_action]:
                            content_label_predict += '_head-in'
                        elif 1 == update_frame_info[ind_task_action]:
                            content_label_predict += '_head-out'
                    if 3 == ind_task_action:
                        if 0 == update_frame_info[ind_task_action]:
                            content_label_predict += '_sit'
                        elif 1 == update_frame_info[ind_task_action]:
                            content_label_predict += '_squat'
                        elif 2 == update_frame_info[ind_task_action]:
                            content_label_predict += '_stand'
                        elif 3 == update_frame_info[ind_task_action]:
                            content_label_predict += '_lie'

                # ------------------- pose block --------------------
                if result_info.predict_type == 'average':
                    if len(result_info.objs[ind_obj].history_pose) > 1:
                        joints_ava = get_pose_ava(result_info.objs[ind_obj].history_pose[:-1])
                        temp = result_info.objs[ind_obj].history_pose[-1].copy()
                        pred = get_pose_new(joints_ava, temp,
                                            result_info.point_dis).astype(np.int32)
                    else:
                        pred = (result_info.objs[ind_obj].history_pose[-1]).astype(np.int32)
                elif result_info.predict_type == 'regression':
                    if len(result_info.objs[ind_obj].history_pose) > 1:
                        joints_predicted = predict_obj_keypoints(result_info.objs[ind_obj].history_pose[:-1])
                        temp = result_info.objs[ind_obj].history_pose[-1].copy()
                        pred = get_pose_new(joints_predicted, temp,
                                            result_info.point_dis).astype(np.int32)
                    else:
                        pred = (result_info.objs[ind_obj].history_pose[-1]).astype(np.int32)
                elif result_info.predict_type == 'combine':
                    if len(result_info.objs[ind_obj].history_pose) > 1:
                        joints_predicted = predict_obj_keypoints(result_info.objs[ind_obj].history_pose[:-1])
                        joints_ava = get_pose_ava(result_info.objs[ind_obj].history_pose[:-1])
                        combine_predict_ava = (np.add(joints_predicted, joints_ava) / 2.0).astype(np.int32)
                        temp = result_info.objs[ind_obj].history_pose[-1].copy()
                        pred = get_pose_new(combine_predict_ava, temp,
                                            result_info.point_dis).astype(np.int32)
                    else:
                        pred = (result_info.objs[ind_obj].history_pose[-1]).astype(np.int32)
                conf_array = np.array(result_info.objs[ind_obj].history_pose_conf)
                chosen_lines = np.logical_and(
                    (conf_array[:, result_info.point_set[:, 0]] > result_info.show_threshold).any(axis=0),
                    (conf_array[:, result_info.point_set[:, 1]] > result_info.show_threshold).any(axis=0))
                high_conf_ind = np.where((conf_array > result_info.show_threshold).any(axis=0))[0]
                draw_skeleton(frame_resize, pred, chosen_lines, result_info.point_set, high_conf_ind)
            content_label_predict_split = content_label_predict.split('_')

            for ind_txt in range(len(content_label_predict_split)):
                cv2.putText(img=frame_resize, text=content_label_predict_split[ind_txt],
                            org=(location_label_predict[0], location_label_predict[1] + ind_txt * 50),
                            fontFace=FONT_LABEL,
                            fontScale=FONT_SCALE, color=COLOR_LABEL, thickness=LINE_WIDTH_LABEL)
                ###show_demo warning
                if 'hand-out' == content_label_predict_split[ind_txt] or 'head-out' == content_label_predict_split[
                    ind_txt]:
                    frame_resize = stick_warning(frame_resize, 'res_warning.jpg')

    # ------------------- objs loss block -------------------
    if result_info.num_objs_loss > result_info.thresh_num_objs_loss:
        content_label_predict = 'miss obj'
        cv2.putText(img=frame_resize, text=content_label_predict, org=(0, 50), fontFace=FONT_LABEL,
                    fontScale=FONT_SCALE, color=COLOR_LABEL, thickness=LINE_WIDTH_LABEL)
    else:
        content_label_predict = 'not miss obj'
        cv2.putText(img=frame_resize, text=content_label_predict, org=(0, 50), fontFace=FONT_LABEL,
                    fontScale=FONT_SCALE, color=COLOR_LABEL, thickness=LINE_WIDTH_LABEL)


def draw_skeleton(img, pred, chosen_points, points, high_conf_points):
    mid_01 = (((pred[0][0] + pred[0][1]) / 2).astype(int) if chosen_points[0] else None)
    mid_56 = (((pred[0][5] + pred[0][6]) / 2).astype(int) if chosen_points[4] else None)
    mid_29 = (((pred[0][2] + pred[0][9]) / 2).astype(int) if chosen_points[2] else None)
    if mid_29 is not None and mid_56 is not None:
        cv2.line(img, tuple(mid_29), tuple(mid_56), (100, 100, 100), thickness=4)
    if mid_01 is not None:
        cv2.circle(img, tuple(mid_01), 10, (100, 100, 100), -1)
        if mid_29 is not None:
            cv2.line(img, tuple(mid_01), tuple(mid_29), (100, 100, 100), thickness=4)
    for ind, line in enumerate(points[1:]):
        if not chosen_points[ind + 1]:
            continue
        else:
            cv2.line(img, tuple(pred[0][line[0]]), tuple(pred[0][line[1]]), (100, 100, 100), thickness=4)
    for point_ind in high_conf_points:
        if point_ind == 0 or point_ind == 1:
            continue
        else:
            cv2.circle(img, tuple(pred[0][point_ind]), 10, (100, 100, 100), -1)


if __name__ == '__main__':
    print(model_info)
