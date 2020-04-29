import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod
from mot.utils import Registry
from mot.tracklet import Tracklet
from mot.prediction import Prediction

__all__ = ['Predictor', 'PREDICTOR_REGISTRY', 'build_predictor']

PREDICTOR_REGISTRY = Registry('predictors')


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.predict(args)

    @abstractmethod
    def initiate(self, tracklets: List[Tracklet]) -> None:
        """
        Initiate tracklets' states that are used by the predictor.
        """
        pass

    @abstractmethod
    def update(self, tracklets: List[Tracklet]) -> None:
        """
        Update tracklets' states that are used by the predictor.
        """
        pass

    @abstractmethod
    def predict(self,args) -> List[Prediction]:
        """
        Predict state in the following time step.

        Args:
            tracklets: A list of tracklet objects. The tracklets that require motion prediction.

        Returns:
            A list of Prediction objects corresponding to the tracklets.
        """
        pass


def build_predictor(cfg):
    if cfg is not None:
        return PREDICTOR_REGISTRY.get(cfg.type)(**(cfg.to_dict()))
    else:
        return None
