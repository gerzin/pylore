"""
This module contains utility functions for defining distance functions
to be used by LORE's fitness function
"""
import numpy as np


class LOREDistance:
    def __init__(self, categorical_mask, **kwargs):
        """ """
        if isinstance(categorical_mask, np.ndarray):
            self.categorical_mask = categorical_mask.astype(bool)
        elif isinstance(categorical_mask, list):
            self.categorical_mask = np.array(categorical_mask, dtype=bool)
        else:
            raise TypeError("Only lists and NumPy arrays are supported.")

        if np.all(self.categorical_mask):
            self.__call__ = self.simple_match
        elif not np.any(self.categorical_mask):
            self.__call__ = self.normalized_eucliden
        else:
            m = len(self.categorical_mask)
            tot_feat = len(self.categorical_mask)
            cat_fet = np.sum(self.categorical_mask)
            num_feat = tot_feat - cat_fet

            self.cat_weight = tot_feat / m
            self.num_weight = num_feat / m

    @classmethod
    def simple_match(cls, x, y):
        return np.sum(x != y)

    @classmethod
    def normalized_eucliden(cls, x, y):
        return np.linalg.norm(x - y)

    def __call__(self, x, y):
        cat_mask = self.categorical_mask
        num_mask = ~cat_mask
        return self.cat_weight * self.simple_match(
            x[cat_mask], y[cat_mask]
        ) + self.num_weight * self.normalized_eucliden(
            x[num_mask], y[num_mask]
        )
