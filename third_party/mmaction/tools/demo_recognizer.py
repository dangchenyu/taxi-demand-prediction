import cv2
import mmcv
import torch
import random
import argparse
import numpy as np
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter, collate
from mmaction.models import build_recognizer


def ramdom_sample(images, num_segments):
    total_images = len(images)
    image_inds = []
    segment_length = int(total_images / num_segments)
    for i in range(num_segments):
        image_inds.append(random.randint(segment_length * i, segment_length * i + segment_length) - 1)
    return images[image_inds]


def inference(model, images):
    result = model([1], None, return_loss=False, img_group_0=torch.Tensor(images))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('video_path', help='path to test video')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(args.video_path)
    cap = cv2.VideoCapture(args.video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video length: ', frame_count)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_count - 1, 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frame_count - 1 and ret):
        ret, frame = cap.read()
        buf[fc] = cv2.resize(frame, (224, 224))
        fc += 1
    print('Video loaded')

    cap.release()

    buf = ramdom_sample(buf, 3)
    buf = buf.transpose((0, 3, 1, 2))
    buf = np.expand_dims(buf, 0)
    buf = buf.astype(np.float32) - 128

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    model = build_recognizer(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, strict=True)

    model.eval()
    outputs = inference(model, buf)

    pred = np.argmax(outputs, axis=1)[0]
    if hasattr(cfg, 'class_names'):
        print('Class #', pred, ': ', cfg.class_names[pred])
    else:
        print('Class #', pred)


if __name__ == '__main__':
    main()
