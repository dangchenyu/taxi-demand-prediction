import numpy as np
from .matcher import Matcher
from scipy import optimize


class HungarianMatcher(Matcher):
    def __init__(self, metric, sigma):
        super().__init__(metric)
        self.sigma=sigma
    def __call__(self, tracklets, detection_features):
        similarity_matrix = self.metric(tracklets, detection_features)
        row_ind, col_ind = optimize.linear_sum_assignment(1 - similarity_matrix)
        valid_inds = [similarity_matrix[row_ind[i], col_ind[i]] > self.sigma for i in range(len(row_ind))]
        row_ind = row_ind[valid_inds]
        col_ind = col_ind[valid_inds]
        return row_ind, col_ind


