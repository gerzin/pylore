"""
This module contains utility functions for defining distance functions
to be used by LORE's fitness function
"""
from abc import ABC, abstractmethod
import numpy as np


class AbstractDistance(ABC):
    @abstractmethod
    def __call__(self, x, y, *args, **kwargs):
        pass


class EuclideanDistance(AbstractDistance):
    """
    Computes the Euclidean Distance between two vectors.
    """

    def __call__(self, x: np.array, y: np.array, *args, **kwargs):
        return np.linalg.norm(x - y)


class SimpleMatchDistance(AbstractDistance):
    """ """

    def __call__(self, x, y, *args, **kwargs):
        dist = 0
        for a, b in zip(x, y):
            if a != b:
                dist += 1
        return dist


__PREDEFINED_DISTANCES = {
    "euclidean": EuclideanDistance,
    "simplematch": SimpleMatchDistance,
}


def dist_from_str(distance: str):
    try:
        return __PREDEFINED_DISTANCES[distance]
    except KeyError:
        raise KeyError(f"{distance} not found among the predefined distances")
