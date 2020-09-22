import torch
from .detect import Detector, Detection
import numpy as np
from detectors.detector_factory import detector_factory
from third_party.CenterNet.src.lib.detectors.ctdet import CtdetDetector
from mot.utils.box import iob

class Centernet(Detector):
    def __init__(self, config, conf_threshold=0.5):
        super(Centernet).__init__()
        self.conf_thres = config.vis_thresh
        self.detector = detector_factory[config.task](config)


    def __call__(self, img):
        raw_result = self.detector.run(img)
        result = raw_result['results'][1][np.where(raw_result['results'][1][:,4] > self.conf_thres)]
        result = result[np.where((result[:,2]- result[:,0])< (result[:,3]- result[:,1]))]#reduce FP h>w
        result = result[np.where((result[:,3]- result[:,1])>115)]
        result = result[np.where((result[:,3]- result[:,1])<350)]
        result = result[np.where(result[:,0]>50)]
        #reduce FP h>50
        #reduce FP h>50
        inds_to_delete = []
        for i in range(len(result)):
            for j in range(len(result)):
                if i != j and iob(result[i], result[j]) > 0.9:
                    inds_to_delete.append(j)

        result = [line.tolist() for line in result]
        inds_to_delete.sort(reverse=True)
        for ind in inds_to_delete:
            result.pop(ind)
        return [Detection(line[:4], line[4]) for line in result]
