import os
import cv2
# import mmcv
import torch
import random
import argparse
import mot.utils
import mot.detect
import mot.encode
import mot.metric
import mot.predict
import numpy as np
import torchvision
import mot.associate
from third_party.TSM import MobileNetV2, GroupNormalize, ToTorchFormatTensor
from mot.tracker import Tracker
from mot.tracklet import Tracklet
from regressor import Box_reg
from PIL import Image
# from mmcv.runner import load_checkpoint
# from mmaction.models import build_recognizer
import time
from opts import opts
from torchvision import transforms


class IoUTracker(Tracker):
    def __init__(self, detector, sigma_conf=0.4):
        metric = mot.metric.IoUMetric(use_prediction=True)
        encoder = mot.encode.ImagePatchEncoder(resize_to=(64, 64))
        matcher = mot.associate.HungarianMatcher(metric, sigma=0.1)
        predictor = mot.predict.KalmanPredictor()
        # predictor=None
        super().__init__(detector, [encoder], matcher, predictor)
        self.sigma_conf = sigma_conf


class Tracktor(Tracker):
    def __init__(self, detector, sigma_active=0.5, lambda_active=0.6, lambda_new=0.3):
        iou_metric = mot.metric.IoUMetric(use_prediction=True)
        iou_matcher = mot.associate.GreedyMatcher(iou_metric, sigma=0.5)
        encoder = mot.encode.ImagePatchEncoder(resize_to=(224, 224))

        matcher = iou_matcher
        predictor = mot.predict.MMTwoStagePredictor(detector)
        self.sigma_active = sigma_active
        self.lambda_active = lambda_active
        self.lambda_new = lambda_new
        super().__init__(detector, [encoder], matcher, predictor)

    def update(self, row_ind, col_ind, detections, detection_features):
        """
        Update the tracklets.
        :param row_ind: A list of integers. Indices of the matched tracklets.
        :param col_ind: A list of integers. Indices of the matched detections.
        :param detection_boxes: A list of Detection objects.
        :param detection_features: The features of the detections. It can be any form you want.
        """
        # Update tracked tracklets' features
        for i in range(len(row_ind)):
            tracklet = self.tracklets_active[row_ind[i]]
            tracklet.update(self.frame_num, tracklet.prediction,
                            {'box': tracklet.prediction.box, **detection_features[col_ind[i]]})

        # Deal with unmatched tracklets
        for i, tracklet in enumerate(self.tracklets_active):
            if tracklet.prediction.score < self.sigma_active:
                if tracklet.fade():
                    self.kill_tracklet(tracklet)

        # Kill tracklets with lower scores using NMS
        tracklets_to_kill = []
        for i, tracklet in enumerate(self.tracklets_active):
            ious = mot.utils.box.iou(tracklet.prediction.box, [t.prediction.box for t in self.tracklets_active])
            overlapping_boxes = np.argwhere(ious > self.lambda_active)
            for j in overlapping_boxes:
                if i == j[0]:
                    continue
                else:
                    if tracklet.prediction.score >= self.tracklets_active[j[0]].prediction.score:
                        if self.tracklets_active[j[0]] not in tracklets_to_kill:
                            tracklets_to_kill.append(self.tracklets_active[j[0]])
                    else:
                        if tracklet not in tracklets_to_kill:
                            tracklets_to_kill.append(tracklet)
                        break
        for tracklet in tracklets_to_kill:
            self.kill_tracklet(tracklet)

        # Update tracklets
        for tracklet in self.tracklets_active:
            tracklet.last_detection.box = tracklet.prediction.box

        # Remove matched detections
        detections_to_remove = []
        for i, detection in enumerate(detections):
            if i not in col_ind:
                for tracklet in self.tracklets_active:
                    if mot.utils.box.iou(detection.box, tracklet.last_detection.box) > self.lambda_new:
                        detections_to_remove.append(detection)
                        break
            else:
                detections_to_remove.append(detection)
        for detection in detections_to_remove:
            detections.remove(detection)

        # Initiate new tracklets
        for i, detection in enumerate(detections):
            new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=1)
            self.add_tracklet(new_tracklet)
            self.predictor.initiate([new_tracklet])


class Taxi_reg:
    def __init__(self, checkpoint):
        self.model = Box_reg()
        sd = torch.load(checkpoint)
        self.model.load_state_dict(sd)
        self.model.eval()

    def __call__(self, img):
        patch_resize = cv2.resize(img, (32, 32))
        img_pil = Image.fromarray(patch_resize).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        input = transform(img_pil)
        input = torch.unsqueeze(input, 0)

        output = self.model(input)
        return output


class TSM:
    def __init__(self, checkpoint, num_segments):
        self.model = MobileNetV2(n_class=2)
        self.num_segments = num_segments
        if checkpoint is not None:
            sd = torch.load(checkpoint)
            sd = sd['state_dict']
            model_dict = self.model.state_dict()
            if isinstance(self.model, MobileNetV2):

                for k in list(sd.keys()):
                    if 'base_model' in k:
                        sd[k.replace('base_model.', '')] = sd.pop(k)
                for k in list(sd.keys()):
                    if 'module' in k:
                        sd[k.replace('module.', '')] = sd.pop(k)

                for k in list(sd.keys()):
                    if '.net' in k:
                        sd[k.replace('.net', '')] = sd.pop(k)
                for k in list(sd.keys()):
                    if 'new_fc' in k:
                        print(sd[k])
                        sd[k.replace('new_fc', 'classifier')] = sd.pop(k)
            model_dict.update(sd)
            self.model.load_state_dict(model_dict)

        self.model.eval()

    def __call__(self, images, *shift_buffer):
        # images = np.array(images)
        # images = images.transpose((0, 3, 1, 2))
        # images = np.expand_dims(images, 0)
        # images = images.astype(np.float32) - 128

        return self.model(images, *shift_buffer)


def process_frame():
    transform = torchvision.transforms.Compose([
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 14, (int(width), int(height)))
    else:
        class MuteVideoWriter():
            def write(self, *args, **kwargs):
                pass

            def release(self):
                pass

        return MuteVideoWriter()


def track_only(tracker, args):
    if os.path.isdir(args.video_path):
        path_list = os.listdir(args.video_path)
        num = 0
        confirmed_num = 45
        for video in path_list:
            video_base_name = os.path.splitext(video)[0]
            capture = cv2.VideoCapture(os.path.join(args.video_path, video))

            # result_writer = open(args.save_result + video_base_name + '_tracked_result.txt', 'w+')

            count = 0
            id_pool = []
            tracker.clear()
            while True:
                ret, frame = capture.read()
                print('frame num', count)
                if not ret:
                    break
                if count < 46800:
                    count += 1
                    continue
                if 19800 < count < 46800:
                    result_writer = open(args.save_result + '1-5' + '_tracked_result.txt', 'a+')
                elif 46800 < count < 75000:
                    result_writer = open(args.save_result + '6-10' + '_tracked_result.txt', 'a+')
                elif 75600 < count < 102600:
                    result_writer = open(args.save_result + '11-15' + '_tracked_result.txt', 'a+')
                elif 102600 < count:
                    result_writer = open(args.save_result + '16-20' + '_tracked_result.txt', 'a+')
                else:
                    result_writer = open(args.save_result + 'others' + '_tracked_result.txt', 'a+')

                tracker.tick(frame)
                frame = mot.utils.visualize_snapshot(frame, tracker)

                for tracklet in tracker.tracklets_active:
                    if tracklet.is_confirmed() and tracklet.is_detected() and len(
                            tracklet.detection_history) >= confirmed_num:
                        # confirm frame num
                        # if tracklet.last_detection.box[3] - tracklet.last_detection.box[1] > 70:
                        detection_history_array = np.array(
                            [i[1] for i in tracklet.detection_history[-confirmed_num + 1:]])
                        ave_dete = np.sum(detection_history_array, axis=0) / (confirmed_num - 1)
                        if (ave_dete[3] - ave_dete[1]) > (
                                ave_dete[2] - ave_dete[0]) * 2 and (
                                ave_dete[3] - ave_dete[1]) > 110:
                            if tracklet.id not in id_pool:
                                id_pool.append(tracklet.id)
                                print(len(id_pool))
                            if tracklet.if_first:
                                for index, box_item in enumerate(tracklet.detection_history[-confirmed_num + 1:]):

                                    count_new = count - confirmed_num + 1 + index + 1
                                    print("____", count_new)
                                    if index == 0:
                                        continue
                                    result_writer.write(
                                        '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(
                                            count_new, tracklet.id,
                                            box_item[1][0],
                                            box_item[1][1],
                                            box_item[1][2] - box_item[1][0],
                                            box_item[1][3] - box_item[1][1]
                                        ))
                                tracklet.if_first = False
                            else:
                                print('after', count)
                                result_writer.write(
                                    '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(
                                        count, tracklet.id,
                                        tracklet.last_detection.box[0],
                                        tracklet.last_detection.box[1],
                                        tracklet.last_detection.box[2] - tracklet.last_detection.box[0],
                                        tracklet.last_detection.box[3] - tracklet.last_detection.box[1]
                                    ))

                cv2.imshow('Demo', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

                count += 1

                result_writer.close()

    else:
        capture = cv2.VideoCapture(args.video_path)
        count = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame = frame[80:, :640]

            count += 1

            # if count<72:
            #     continue
            # if 72<count<83:
            #     cv2.imwrite('/home/rvlab/Desktop/frames_temp/{}.jpg'.format(count),frame)
            tracker.tick(frame)
            frame = mot.utils.visualize_snapshot(frame, tracker)

            cv2.imshow('Demo', frame)
            key = cv2.waitKey(0)
            if key == 27:
                break


def track_and_recognize(tracker, recognizer, regressor, args):
    if os.path.isdir(args.video_path):
        spot_list = os.listdir(args.video_path)
        for spot in spot_list:
            video_list = os.listdir(os.path.join(args.video_path, spot))
            for video in video_list:
                if video in ['M19040111460800061.mp4', 'M19040212172600021.mp4', 'M19031611152900021.mp4',
                             'M19032415180200611.mp4',
                             'M19032716550100221.mp4',
                             'M19032614082200081.mp4',
                             'M19032814123800641.mp4',
                             'M19032908130200261.mp4',
                             'M19032908130200261.mp4',
                             'M19040411475100711.mp4',
                             'M19031618103700211.mp4',
                             'M19032815265400351.mp4'
                             ]:
                    capture = cv2.VideoCapture(os.path.join(args.video_path, spot, video))

                    video_writer = get_video_writer(os.path.join(args.save_video, 'analyzed' + video), 640,
                                                    640)
                    count = 0
                    # shift_buffer = [torch.zeros([1, 3, 16, 8]),
                    #                 torch.zeros([1, 4, 8, 4]),
                    #                 torch.zeros([1, 4, 8, 4]),
                    #                 torch.zeros([1, 8, 4, 2]),
                    #                 torch.zeros([1, 8, 4, 2]),
                    #                 torch.zeros([1, 8, 4, 2]),
                    #                 torch.zeros([1, 12, 4, 2]),
                    #                 torch.zeros([1, 12, 4, 2])]
                    shift_buffer = [torch.zeros([1, 3, 16, 16]),
                                    torch.zeros([1, 4, 8, 8]),
                                    torch.zeros([1, 4, 8, 8]),
                                    torch.zeros([1, 8, 4, 4]),
                                    torch.zeros([1, 8, 4, 4]),
                                    torch.zeros([1, 8, 4, 4]),
                                    torch.zeros([1, 12, 4, 4]),
                                    torch.zeros([1, 12, 4, 4]),
                                    torch.zeros([1, 20, 2, 2]),
                                    torch.zeros([1, 20, 2, 2])]

                    while True:
                        start_time = time.time()
                        # if count<24000:
                        #     count+=1
                        #     ret, frame = capture.read()

                        # continue
                        ret, frame = capture.read()

                        if not ret:
                            break
                        frame = frame[80:, :640]
                        tracker.tick(frame)
                        frame = mot.utils.visualize_snapshot(frame, tracker)

                        # Perform action recognition each second
                        for tracklet in tracker.tracklets_active:
                            if tracklet.is_confirmed() and tracklet.is_detected() and len(
                                    tracklet.feature_history) >= 2:
                                # cv2.imshow('test',tracklet.feature_history[-1][1]['patch'])
                                # cv2.waitKey(0)
                                patch = cv2.cvtColor(tracklet.feature_history[-1][1]['patch'], cv2.COLOR_BGR2RGB)
                                sample = process_frame()(patch)
                                sample = torch.unsqueeze(sample, 0)

                                if tracklet.buffer == []:
                                    tracklet.buffer.append(shift_buffer)
                                prediction, *out_buffer = recognizer(sample, *tracklet.buffer[-1])
                                tracklet.buffer.append(out_buffer)
                                tracklet.past_feat_action.append(prediction.detach().cpu())
                                if len(tracklet.past_feat_action) >= 8:
                                    feat_for_predict = tracklet.past_feat_action[-8:]
                                    avg_logit = sum(feat_for_predict)

                                    print(feat_for_predict, avg_logit)
                                    action = np.argmax(avg_logit, axis=1)[0]
                                    temp = 0
                                else:
                                    action = None
                                if action == 0:
                                    box = tracklet.last_detection.box
                                    frame = cv2.putText(frame, 'walking', (int(box[0]), int(box[1]) - 25),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                                    frame = cv2.putText(frame, 'prob: 0', (int(box[0]), int(box[1]) - 4),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                                elif action == 1:
                                    wait_prob = regressor(patch[0:int(patch.shape[0] / 2), :])
                                    tracklet.wait_prob.append(wait_prob)
                                    if len(tracklet.wait_prob) >= 3:

                                        tracklet.last_wait_prob = ((sum(tracklet.wait_prob) - min(
                                            tracklet.wait_prob) - max(tracklet.wait_prob)) / (
                                                                           len(
                                                                               tracklet.wait_prob) - 2)).detach().cpu().numpy()[
                                            0][0]
                                    else:
                                        tracklet.last_wait_prob = \
                                        (sum(tracklet.wait_prob) / len(tracklet.wait_prob)).detach().cpu().numpy()[0][0]
                                    box = tracklet.last_detection.box
                                    frame = cv2.putText(frame, 'standing', (int(box[0]), int(box[1]) - 25),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=2)
                                    frame = cv2.putText(frame, 'prob: %.2f' % tracklet.last_wait_prob,
                                                        (int(box[0]), int(box[1]) - 4),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                                        0, 255 * tracklet.last_wait_prob,
                                                        255 * (1 - tracklet.last_wait_prob)),
                                                        thickness=2)

                        end_time = time.time()
                        print(end_time - start_time, 'total people', len(tracker.tracklets_active))
                        cv2.imshow('Demo', frame)
                        count += 1
                        video_writer.write(frame)
                        key = cv2.waitKey(1)
                        if key == 27:
                            break

                    video_writer.release()
    else:
        capture = cv2.VideoCapture(args.video_path)

        # video_writer = get_video_writer(os.path.join(args.save_video, 'analyzed' + os.path.basename(args.video_path)), 640,
        #                                 640)
        count = 0
        # shift_buffer = [torch.zeros([1, 3, 16, 8]),
        #                 torch.zeros([1, 4, 8, 4]),
        #                 torch.zeros([1, 4, 8, 4]),
        #                 torch.zeros([1, 8, 4, 2]),
        #                 torch.zeros([1, 8, 4, 2]),
        #                 torch.zeros([1, 8, 4, 2]),
        #                 torch.zeros([1, 12, 4, 2]),
        #                 torch.zeros([1, 12, 4, 2])]
        shift_buffer = [torch.zeros([1, 3, 16, 16]),
                        torch.zeros([1, 4, 8, 8]),
                        torch.zeros([1, 4, 8, 8]),
                        torch.zeros([1, 8, 4, 4]),
                        torch.zeros([1, 8, 4, 4]),
                        torch.zeros([1, 8, 4, 4]),
                        torch.zeros([1, 12, 4, 4]),
                        torch.zeros([1, 12, 4, 4]),
                        torch.zeros([1, 20, 2, 2]),
                        torch.zeros([1, 20, 2, 2])]

        while True:
            start_time = time.time()
            reg_end=0
            act_end=0
            reg_start=0
            if count<150:
                count+=1
                ret, frame = capture.read()

                continue
            ret, frame = capture.read()
            ret, frame = capture.read()

            if not ret:
                break
            # frame = frame[80:, :640]
            tracker.tick(frame)
            frame = mot.utils.visualize_snapshot(frame, tracker)

            # Perform action recognition each second
            act_start=time.time()
            for tracklet in tracker.tracklets_active:
                if tracklet.is_confirmed() and tracklet.is_detected() and len(
                        tracklet.feature_history) >= 2:
                    # cv2.imshow('test',tracklet.feature_history[-1][1]['patch'])
                    # cv2.waitKey(0)
                    patch = cv2.cvtColor(tracklet.feature_history[-1][1]['patch'], cv2.COLOR_BGR2RGB)
                    sample = process_frame()(patch)
                    sample = torch.unsqueeze(sample, 0)

                    if tracklet.buffer == []:
                        tracklet.buffer.append(shift_buffer)
                    prediction, *out_buffer = recognizer(sample, *tracklet.buffer[-1])
                    tracklet.buffer.append(out_buffer)
                    tracklet.past_feat_action.append(prediction.detach().cpu().numpy())
                    if len(tracklet.past_feat_action) >= 10:
                        # walk_ind=np.where(np.squeeze(tracklet.past_feat_action[- 8:],1)[:,0]>0)
                        # if len(walk_ind[0])>=(8/2):
                        #     action=0
                        # else:
                        #     action=1
                        feat_for_predict = tracklet.past_feat_action[-10:]
                        avg_logit = sum(feat_for_predict)

                        # print(feat_for_predict, avg_logit,tracklet.id)
                        action = np.argmax(avg_logit, axis=1)[0]
                        temp = 0
                    else:
                        action = None
                    act_end=time.time()
                    if action == 0:
                        box = tracklet.last_detection.box
                        frame = cv2.putText(frame, 'walking', (int(box[0]), int(box[1]) - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                        frame = cv2.putText(frame, 'prob: 0', (int(box[0]), int(box[1]) - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                    elif action == 1:
                        reg_start=time.time()
                        wait_prob = regressor(patch[0:int(patch.shape[0] / 2), :])
                        reg_end=time.time()
                        tracklet.wait_prob.append(wait_prob)
                        if len(tracklet.wait_prob) >= 3:

                            tracklet.last_wait_prob = ((sum(tracklet.wait_prob) - min(
                                tracklet.wait_prob) - max(tracklet.wait_prob)) / (
                                                               len(
                                                                   tracklet.wait_prob) - 2)).detach().cpu().numpy()[
                                0][0]
                        else:
                            tracklet.last_wait_prob = \
                                (sum(tracklet.wait_prob) / len(tracklet.wait_prob)).detach().cpu().numpy()[0][0]
                        box = tracklet.last_detection.box
                        frame = cv2.putText(frame, 'standing', (int(box[0]), int(box[1]) - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=2)
                        frame = cv2.putText(frame, 'prob: %.2f' % tracklet.last_wait_prob,
                                            (int(box[0]), int(box[1]) - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (
                                                0, 255 * tracklet.last_wait_prob,
                                                255 * (1 - tracklet.last_wait_prob)),
                                            thickness=2)
            if len(tracker.tracklets_to_kill)!=0:
                temp=0
            end_time = time.time()
            print(end_time - start_time, 'total people', len(tracker.tracklets_active))
            print('act_time:{},reg_time:{}'.format(act_end-act_start,reg_end-reg_start))

            cv2.imshow('Demo', frame)
            count += 1
            # video_writer.write(frame)
            key = cv2.waitKey(0)
            if key == 27:
                break

        # video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['track', 'action'], help='track only or recognize action too')
    parser.add_argument('-d', '--detector_config', help='test config file of object detector')
    parser.add_argument('-dw', '--detector_checkpoint', required=True, help='checkpoint file of object detector')
    parser.add_argument('-rc', '--regressor_ck', required=True, help='checkpoint file of regressor')

    parser.add_argument('-r', '--recognizer_config', required=False, help='test config file of TSN action recognizer')
    parser.add_argument('-rw', '--recognizer_checkpoint', required=False,
                        help='checkpoint file of TSN action recognizer')
    parser.add_argument('-t', '--tracker', default='tracktor', choices=['tracktor', 'ioutracker'])
    parser.add_argument('-i', '--video_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('-o', '--save_video',
                        default='/home/rvlab/Documents/DRDvideos/may_contain_passemger/track_result/', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('-s', '--save_result',
                        default='/home/rvlab/Documents/DRDvideos/campus/track_result/', required=False,
                        help='Path to the output track result file. Leave it blank to disable.')
    parser.add_argument('--detector', required=True, default='mmdetection',
                        help='choose detector(mmdetection/centernet/acsp)')
    parser.add_argument('--recognizer', required=True, default='TSN', help='choose action recognizer(TSN/TSM)')
    parser.add_argument('--num_segments', default=6, help='set segments num for action part')
    args = parser.parse_args()
    if args.detector == 'mmdetection':
        detector = mot.detect.MMDetector(args.detector_config, args.detector_checkpoint)
    if args.detector == 'centernet':
        opt = opts()
        opt.init(args.detector_checkpoint)
        detector = mot.detect.Centernet(opt)
    if args.tracker == 'tracktor':
        tracker = Tracktor(detector, sigma_active=0.6)
    elif args.tracker == 'ioutracker':
        tracker = IoUTracker(detector, sigma_conf=0.3)
    else:
        raise AssertionError('Unknown tracker')

    if args.mode == 'track':
        track_only(tracker, args)
    elif args.mode == 'action':

        if args.recognizer == 'TSM':
            recognizer = TSM(args.recognizer_checkpoint, args.num_segments)
        regressor = Taxi_reg(args.regressor_ck)
        track_and_recognize(tracker, recognizer, regressor, args)
