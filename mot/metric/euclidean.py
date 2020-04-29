from .metric import Metric
import numpy as np
import mot.utils.box
import mot.utils.debug
class EuclideanMetric(Metric):
    """
    An affinity metric that only considers the euclidean of tracklets' box and detected box.
    """

    def __init__(self, use_prediction=False):
        super(EuclideanMetric).__init__()
        self.encoding = 'box'
        self.use_prediction = use_prediction

    def __call__(self, tracklets, detection_features):

        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                if self.use_prediction:
                    matrix[i][j] = mot.utils.box._pdist(tracklets[i].prediction.box, detection_features[j][self.encoding])
                else:
                    matrix[i][j] = mot.utils.box._pdist(tracklets[i].last_detection.box,
                                                     detection_features[j][self.encoding])

        mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix