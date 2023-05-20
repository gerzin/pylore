"""
This module contains the implementation of the function that creates
a synthetic neighborhood of an instance using a genetic algorithm.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def mutator(matrix, num_features, categorical_mask):
    num_samples, num_features_orig = matrix.shape

    # Generate random indices to select features to change
    change_indices = np.random.choice(
        num_features_orig, size=num_features, replace=False
    )

    # Create a copy of the matrix to store the mutated samples
    mutated_matrix = np.copy(matrix)

    for sample_idx in range(num_samples):
        for feature_idx in change_indices:
            # Check if the feature is categorical or continuous
            if categorical_mask[feature_idx]:
                # If categorical, randomly select a new value
                # based on the feature probability
                new_value = np.random.choice(np.unique(matrix[:, feature_idx]))
            else:
                # If continuous, generate a new value
                # from the feature distribution
                feature_values = matrix[:, feature_idx]
                feature_mean = np.mean(feature_values)
                feature_std = np.std(feature_values)
                new_value = np.random.normal(
                    loc=feature_mean, scale=feature_std
                )

            # Assign the new value to the mutated sample
            mutated_matrix[sample_idx, feature_idx] = new_value

    return mutated_matrix


class Mutator:
    def __init__(self, pm, K, categorical_mask=None):
        self.pm_ = pm
        self.K_ = K
        self.categorical_mask = categorical_mask

    def fit(self, data):
        pass

    def __call__(self, *args, **kwargs):
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
