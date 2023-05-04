from pylore.blackbox import AbstractBlackBoxWrapper
from pylore.distances import EuclideanDistance, dist_from_str
from sklearn.tree import DecisionTreeClassifier
from typing import Union, Callable
import numpy as np


class LORE:
    def __init__(
        self,
        bb: AbstractBlackBoxWrapper,
        neighbors: int,
        distance: Union[
            Callable[[np.array, np.array], np.number], str
        ] = EuclideanDistance,
        **kwargs
    ):
        self.bb_ = bb
        self.neighbors_ = neighbors
        self.distance_ = (
            distance if type(distance) is not str else dist_from_str(distance)
        )

        # default values from the paper
        self.generations_ = kwargs.get("generations", 10)
        self.crossover_prob_ = kwargs.get("crossover_prob", 0.5)
        self.mutation_prob_ = kwargs.get("mutation_prob", 0.2)

        # instantiate the classifier
        self.random_state_ = kwargs.get("random_state")
        self.clf_ = DecisionTreeClassifier(random_state=self.random_state_)

    @property
    def black_box(self):
        return self.bb_

    @property
    def random_state(self):
        return self.random_state_

    def __call__(self, x, *args, **kwargs):
        pass
