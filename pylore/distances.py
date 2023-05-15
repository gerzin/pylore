"""
This module contains utility functions for defining distance functions
to be used by LORE's fitness function
"""
from abc import ABC, abstractmethod
import numpy as np


class AbstractDistance(ABC):
    @abstractmethod
    def __call__(self, x, y, mask=None):
        raise NotImplementedError


class EuclideanDistance(AbstractDistance):
    """
    Computes the Euclidean Distance between two vectors.
    """

    def __call__(self, x: np.array, y: np.array, mask=None):
        return np.linalg.norm(x - y)


class SimpleMatchDistance(AbstractDistance):
    """Simple match distance.

    Counts the number of position in which two vectors differ.
    """

    def __call__(self, x, y, mask=None):
        mask = mask if mask else np.full_like(x, True)
        # Get the categorical feature indices
        categorical_indices = np.where(mask)[0]

        # Compute the simple match distance for categorical features
        categorical_distance = np.sum(
            x[categorical_indices] != y[categorical_indices]
        )

        return categorical_distance


__PREDEFINED_DISTANCES = {
    "euclidean": EuclideanDistance,
    "simplematch": SimpleMatchDistance,
}


def dist_from_str(distance: str):
    try:
        return __PREDEFINED_DISTANCES[distance]
    except KeyError:
        raise KeyError(f"{distance} not found among the predefined distances")
