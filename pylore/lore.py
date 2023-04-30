import numpy as np

from pylore.blackbox import AbstractBlackBoxWrapper
from pylore.distances import AbstractDistance
from sklearn import tree

from pylore.genetics import genetic_neighborhood


class LORE:
    def __init__(
        self,
        bb: AbstractBlackBoxWrapper,
        neighbors: int,
        distance: AbstractDistance,
        **kwargs
    ):
        self.bb_ = bb
        self.neighbors_ = neighbors
        self.distance_ = distance

        # default values from the paper
        self.generations_ = kwargs.get("generations", 10)
        self.crossover_prob_ = kwargs.get("crossover_prob", 0.5)
        self.mutation_prob_ = kwargs.get("mutation_prob", 0.2)

        # instantiate the classifier
        self.random_state_ = kwargs.get("random_state")
        self.clf_ = tree.DecisionTreeClassifier(random_state=self.random_state_)

    @property
    def black_box(self):
        return self.bb_

    def __call__(self, x, *args, **kwargs):
        pass
