"""
This module contains the implementation of the function that creates
a synthetic neighborhood of an instance using a genetic algorithm.
"""
from abc import ABC, abstractmethod


class AbstractMutator(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def __call__(self, x, *args, **kwargs):
        pass


class DefaultMutator(AbstractMutator):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def __call__(self, x, *args, **kwargs):
        pass


class AbstractCrossoverer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class TwoPointCrossoverer(AbstractCrossoverer):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
