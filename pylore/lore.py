from numba import njit
import numpy as np

from pylore.blackbox import AbstractBlackBoxWrapper
from pylore.distances import AbstractDistance
from sklearn import tree

from pylore.genetics import genetic_neighborhood


class LORE:
    """

    """

    def __init__(self, bb: AbstractBlackBoxWrapper, neighbors: int, distance: AbstractDistance, **kwargs):
        """

        :param bb:
        :param neighbors:
        :param distance:
        :keyword generations (int, default 10): number of generations for the genetic neighborhood construction.
        :keyword crossover_prob:
        :keyword mutation_prob:
        :keyword random_state (int): Controls the randomness of the estimator.

        """
        self.bb = bb
        self.neighbors = neighbors
        self.distance = distance

        # default values from the paper
        self.generations = kwargs.get('generations', 10)
        self.crossover_prob = kwargs.get('crossover_prob', 0.5)
        self.mutation_prob = kwargs.get('mutation_prob', 0.2)

        # instantiate the classifier
        random_state = kwargs.get('random_state')
        self.clf = tree.DecisionTreeClassifier(random_state=random_state)

    def __call__(self, x, *args, **kwargs):
        """

        :param x: instance to explain
        :param args:
        :param kwargs:
        :return:
        """



        pass


