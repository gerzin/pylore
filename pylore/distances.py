"""
This module contains utility functions for defining distance functions to be used by LORE's fitness function
"""
from abc import ABC, abstractmethod
import numpy as np
from numba import njit


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


class NormalizedEuclideanDistance(AbstractDistance):
    """
    Normalized the vectors and computes the Euclidean Distance between them.
    """

    def __call__(self, x: np.array, y: np.array, *args, **kwargs):
        diff = normalize_vector(x - y)
        return np.linalg.norm(diff)


class SimpleMatchDistance(AbstractDistance):
    """

    """
    def __call__(self, x, y, *args, **kwargs):
        dist = 0
        for (a, b) in zip(x, y):
            if a != b:
                dist += 1
        return dist


class MixedAttributesDistance(AbstractDistance):
    """
    Computes the distance between two instances when these are composed of both categorical and continuous attributes.


    """
    def __init__(self, sep_index: int, **kwargs):
        self.sep_index = sep_index
        self.categ_dist = SimpleMatchDistance()
        self.neucl_dist = NormalizedEuclideanDistance()

    def __call__(self, x, y, *args, **kwargs):
        s = self.sep_index
        categorical = x[:s], y[:s]
        continuous = x[s:], y[:s]
        nelem = len(x)
        return (self.sep_index / nelem) * self.categ_dist(*categorical) + (
                (nelem - self.sep_index) / nelem) * self.neucl_dist(
            *continuous)


# utilities
def normalize_vector(x: np.array):
    norm = np.dot(np.sqrt(x))
    if norm:
        return x / norm
    else:
        return norm


@njit
def numba_simple_match(x, y):
    dist = 0
    for (a, b) in zip(x, y):
        if a != b:
            dist += 1
    return dist
