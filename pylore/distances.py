"""
This module contains utility functions for defining distance functions
to be used by LORE's fitness function
"""
import numpy as np
import pandas as pd
from numba import njit


@njit
def simple_match_distance(x, y):
    """Compute the simple match distance.

    Compute the SMD computed as 1-SMC with SMC=#matching attr./#tot attr.
    """
    return 1 - np.sum(x == y) / len(x)


@njit
def normalized_square_eucludean_distance(x, y, eps=1e-8):
    """Compute the normalized square euclidean distance.

    https://reference.wolfram.com/language/ref/NormalizedSquaredEuclideanDistance.html
    """
    var_x1 = np.var(x, keepdims=True)
    var_x2 = np.var(y, keepdims=True)

    ned_2 = 0.5 * (np.var(x - y, keepdims=True) / (var_x1 + var_x2 + eps))
    return ned_2


class LOREDistance:
    def __init__(self, categorical_mask=None, **kwargs):
        """
        Arguments:
        ---
        * categorical_mask
        """

        if isinstance(categorical_mask, np.ndarray):
            self.categorical_mask = categorical_mask.astype(bool)
        elif isinstance(categorical_mask, list):
            self.categorical_mask = np.array(categorical_mask, dtype=bool)
        elif isinstance(categorical_mask, pd.DataFrame):
            # automatically extract the mask from a DataFrame
            object_cols = categorical_mask.select_dtypes(
                include=["object"]
            ).columns
            self.categorical_mask = categorical_mask.columns.isin(object_cols)
        else:
            raise TypeError(
                "Only lists, NumPy arrays and Dataframes are supported"
            )
        self.numerical_mask = ~self.categorical_mask

        if np.all(self.categorical_mask):
            self.__call__ = lambda self, x, y: simple_match_distance(x, y)
        elif not np.any(self.categorical_mask):
            self.__call__ = (
                lambda self, x, y: normalized_square_eucludean_distance(x, y)
            )
        else:
            m = len(self.categorical_mask)
            tot_feat = len(self.categorical_mask)
            cat_fet = np.sum(self.categorical_mask)
            num_feat = tot_feat - cat_fet

            self.cat_weight = tot_feat / m
            self.num_weight = num_feat / m

    def __call__(self, x, y):
        return self.cat_weight * simple_match_distance(
            x[self.categorical_mask], y[self.categorical_mask]
        ) + self.num_weight * normalized_square_eucludean_distance(
            x[self.numerical_mask], y[self.numerical_mask]
        )
