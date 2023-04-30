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
        diff = x - y
        return np.linalg.norm(diff)


class SimpleMatchDistance(AbstractDistance):
    """ """

    def __call__(self, x, y, *args, **kwargs):
        dist = 0
        for a, b in zip(x, y):
            if a != b:
                dist += 1
        return dist


PREDEFINED_DISTANCES = {
    "euclidean": EuclideanDistance,
    "simplematch": SimpleMatchDistance,
}


def dist_from_str(distance: str):
    try:
        return PREDEFINED_DISTANCES[distance]
    except KeyError:
        raise KeyError(f"{distance} not found in predefined distances")
