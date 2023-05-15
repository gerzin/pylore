from pylore.blackbox import AbstractBlackBox
from sklearn.tree import DecisionTreeClassifier
from typing import Callable
import numpy as np
from pylore.treeutils import extract_decision_rule, extract_counterfactuals


class LORE:
    def __init__(
        self,
        black_box: AbstractBlackBox,
        neighbors: int,
        distance: Callable[[np.array, np.array], np.number],
        **kwargs,
    ):
        """

        Keywords:
        * generations: int -
        * crossover_prob: float -
        * mutation_prob: float -
        * random_state -
        """
        self.bb_ = black_box
        self.neighbors_ = neighbors
        self.distance_ = distance

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
    def distance(self):
        return self.distance_

    @property
    def neighbors(self):
        return self.neighbors_

    @property
    def explainer(self):
        return self.clf_

    def generate_neighbors(self, x, fitness, **kwargs):
        """Generate the genetic neighbors."""

        black_box = kwargs.get("black_box", self.black_box)
        neighbors = kwargs.get("neighbors", self.neighbors)
        generations = kwargs.get("generations", self.generations_)

        population = np.repeat(x[np.newaxis, ...], neighbors, axis=0)

        def evaluate_population(population):
            pass

        for _ in range(generations):
            print(population)
            print(black_box)
            evaluate_population(population)

        return population

    def build_decision_tree(dataset):
        dt = DecisionTreeClassifier()
        dt.fit(dataset)
        return dt

    def __call__(self, x, *args, **kwargs):
        z_eq = self.generate_neighbors(x, lambda x: x)
        z_neq = self.generate_neighbors(x, lambda x: x)
        z = np.concatenate([z_eq, z_neq])
        self.clf_ = self.build_decision_tree(z)
        decision_rule = extract_decision_rule(self.explainer, x, **kwargs)
        counterfactuals = extract_counterfactuals(self.clf_, decision_rule)
        return decision_rule, counterfactuals
