"""
This module contains the implementation of the function that creates a synthetic neighborhood of an instance using a genetic algorithm.
"""
from abc import ABC, abstractmethod

from pylore.blackbox import AbstractBlackBoxWrapper
from typing import Callable
import numpy as np
from numba import njit

from pylore.distances import AbstractDistance


class AbstractMutator(ABC):

    @abstractmethod
    def fit(self, data):
        """
        Extract the empirical distribution from the data in order to be able to mutate some attributes.
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def __call__(self, x, *args, **kwargs):
        pass


class DefaultMutator(AbstractMutator):

    def __init__(self):
        # separator
        pass

    def fit(self, data):
        pass

    def __call__(self, x, *args, **kwargs):
        pass


class AbstractCrossoverer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        p1 = ...
        p2 = ...
        features = np.random.randint(0, p1.shape[0], size=(2,))


class TwoPointCrossoverer(AbstractCrossoverer):
    """

    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        # select two items at random
        p1, p1 = ...
        # swap two random features
        idx = np.random.randint(0, p1.shape[0], size=(2,))
        p1[idx], p2[idx] = p2[idx], p1[idx]
        pass
