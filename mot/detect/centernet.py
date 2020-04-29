import torch
from .detect import Detector, Detection
import numpy as np
from detectors.detector_factory import detector_factory
from third_party.CenterNet.src.lib.detectors.ctdet import CtdetDetector


class Centernet(Detector):
    def __init__(self, config, conf_threshold=0.5):
        super(Centernet).__init__()
        self.conf_thres = config.vis_thresh
        self.detector = detector_factory[config.task](config)


    def __call__(self, img):
        raw_result = self.detector.run(img)
        result = raw_result['results'][1][np.where(raw_result['results'][1][:,4] > self.conf_thres)]
        return [Detection(line[:4], line[4]) for line in result]
