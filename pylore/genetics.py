"""
This module contains the implementation of the function that creates
a synthetic neighborhood of an instance using a genetic algorithm.
"""
import numpy as np
from numba import njit


class Mutator:
    def __init__(self, pm, K):
        self.pm_ = pm
        self.K_ = K

    def fit(self, data):
        pass

    def __call__(self, x, *args, **kwargs):
        pass


@njit(cache=True)
def crossover_numba(population, k):
    n_parents, n_features = population.shape
    new_population = np.zeros((n_parents // 2, n_features))

    # Perform crossover between consecutive parents
    for i in range(0, n_parents, 2):
        parent1 = population[i]
        parent2 = population[i + 1]

        indices = np.random.choice(n_features, size=k, replace=False)

        child = parent1.copy()
        child[indices] = parent2[indices]

        new_population[i // 2] = child

    return new_population


class KPointCrossoverer:
    def __init__(self, pc, K=2, feat_prob=None):
        """ """
        self.pc_ = pc
        self.K_ = K
        self.feat_prob_ = feat_prob

    @property
    def probability(self):
        return self.pc_

    @property
    def K(self):
        return self.K_

    def __call__(self, *args, **kwargs):
        """

        Arguments:

        """
        dataset = args[0]
        len_dataset = dataset.shape[0]
        pc = args[1] if len(args) > 1 else self.pc_
        K = kwargs.get("K", self.K)

        n_parents = int(len_dataset * pc) if pc <= 1 else pc

        assert n_parents <= len_dataset, f"{n_parents = } but {len_dataset = }"

        np.random.shuffle(dataset)
        parents = dataset[:n_parents]
        return crossover_numba(parents, K)
