import os
import cv2
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
from third_party.TSM import MobileNetV2,GroupNormalize,ToTorchFormatTensor
from mot.tracker import Tracker
from mot.tracklet import Tracklet


from opts import opts


class IoUTracker(Tracker):
    def __init__(self, detector, sigma_conf=0.4):
        metric = mot.metric.IoUMetric(use_prediction=False)
        encoder = mot.encode.ImagePatchEncoder(resize_to=(32, 32))
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





class TSM:
    def __init__(self, checkpoint,num_segments):
        self.model = MobileNetV2()
        self.num_segments=num_segments
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
                        sd[k.replace('new_fc', 'classifier')] = sd.pop(k)
            model_dict.update(sd)
            self.model.load_state_dict(model_dict)

        self.model.eval()

    def __call__(self, images):
        # images = np.array(images)
        # images = images.transpose((0, 3, 1, 2))
        # images = np.expand_dims(images, 0)
        # images = images.astype(np.float32) - 128

        return self.model(images)


def ramdom_sample(images, num_segments):
    total_images = len(images)
    image_inds = []
    segment_length = int(total_images / num_segments)
    for i in range(num_segments):
        image_inds.append(random.randint(segment_length * i, segment_length * i + segment_length - 1))
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    transform = torchvision.transforms.Compose([
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)])
    # for ind in image_inds:
    #     cv2.imshow('templates',images[ind])
    #     cv2.waitKey(0)
    images_list=[transform(images[ind]) for ind in image_inds]
    image_tensor=torch.cat(images_list,0)
    return image_tensor


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
        for action_name in path_list:
            if action_name in ['walk']:
                video_list = os.listdir(os.path.join(args.video_path, action_name))
                for video in video_list:
                    video_base_name = os.path.splitext(video)[0]
                    capture = cv2.VideoCapture(os.path.join(args.video_path, action_name, video))
                    video_writer = get_video_writer(os.path.join(args.save_video, action_name, video),
                                                    capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    result_writer = open(os.path.join(args.save_video, action_name, video_base_name + '.txt'), 'w+')

                    count = 0
                    tracker.clear()
                    while True:
                        ret, frame = capture.read()

                        if not ret:
                            break
                        tracker.tick(frame)
                        frame = mot.utils.visualize_snapshot(frame, tracker)

                        # Perform action recognition each second
                        for tracklet in tracker.tracklets_active:
                            if tracklet.is_confirmed() and tracklet.is_detected() and len(
                                    tracklet.feature_history) >= 3:
                                result_writer.write(
                                    '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(
                                        count, tracklet.id,
                                        tracklet.last_detection.box[0],
                                        tracklet.last_detection.box[1],
                                        tracklet.last_detection.box[2] - tracklet.last_detection.box[0],
                                        tracklet.last_detection.box[3] - tracklet.last_detection.box[1]
                                    ))

                        cv2.imshow('Demo', frame)
                        video_writer.write(frame)
                        key = cv2.waitKey(1)
                        if key == 27:
                            break

                        count += 1

                    video_writer.release()
                    result_writer.close()
    else:
        capture = cv2.VideoCapture(args.video_path)
        count = 0
        while True:
            ret, frame = capture.read()
            frame = frame[80:, :640]
            if not ret:
                break
            count += 1

            # if count<72:
            #     continue
            # if 72<count<83:
            #     cv2.imwrite('/home/rvlab/Desktop/frames_temp/{}.jpg'.format(count),frame)
            tracker.tick(frame)
            frame = mot.utils.visualize_snapshot(frame, tracker)

            cv2.imshow('Demo', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break


def track_and_recognize(tracker, recognizer, args):
    capture = cv2.VideoCapture(args.video_path)
    # video_writer = get_video_writer(args.save_video, capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    # capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=0
    while True:

        ret, frame = capture.read()
        if count<900:
            count+=1
            continue
        if not ret:
            break
        frame = frame[80:, :640]
        tracker.tick(frame)
        frame = mot.utils.visualize_snapshot(frame, tracker)

        # Perform action recognition each second
        for tracklet in tracker.tracklets_active:
            if tracklet.is_confirmed() and tracklet.is_detected() and len(tracklet.feature_history) >= args.num_segments:
                samples = ramdom_sample([feature[1]['patch'] for feature in tracklet.feature_history], args.num_segments)
                prediction = recognizer(samples)
                action = np.argmax(prediction.detach().cpu())
                if action == 0:
                    box = tracklet.last_detection.box
                    frame = cv2.putText(frame, 'walking', (int(box[0] + 4), int(box[1]) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
                elif action == 1:
                    box = tracklet.last_detection.box
                    frame = cv2.putText(frame, 'standing', (int(box[0] + 4), int(box[1]) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)

        cv2.imshow('Demo', frame)
        # video_writer.write(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['track', 'action'], help='track only or recognize action too')
    parser.add_argument('-d', '--detector_config',  help='test config file of object detector')
    parser.add_argument('-dw', '--detector_checkpoint', required=True, help='checkpoint file of object detector')
    parser.add_argument('-r', '--recognizer_config', required=False, help='test config file of TSN action recognizer')
    parser.add_argument('-rw', '--recognizer_checkpoint', required=False,
                        help='checkpoint file of TSN action recognizer')
    parser.add_argument('-t', '--tracker', default='tracktor', choices=['tracktor', 'ioutracker'])
    parser.add_argument('-i', '--video_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('-o', '--save_video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('-s', '--save_result', default='', required=False,
                        help='Path to the output track result file. Leave it blank to disable.')
    parser.add_argument('--detector', required=True, default='mmdetection',
                        help='choose detector(mmdetection/centernet/acsp)')
    parser.add_argument('--recognizer', required=True, default='TSN', help='choose action recognizer(TSN/TSM)')
    parser.add_argument('--num_segments', default=4, help='set segments num for action part')
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
            recognizer = TSM(args.recognizer_checkpoint,args.num_segments)
        track_and_recognize(tracker, recognizer, args)
