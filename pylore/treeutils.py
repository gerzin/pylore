# flake8: noqa
"""
This module contains utilities for extracting decision rules
and counterfactuals from a DecisionTreeClassifier.
"""
from sklearn.tree import DecisionTreeClassifier, _tree
from typing import Union


def extract_decision_rule(
    dt: Union[DecisionTreeClassifier, _tree.Tree], x, features_names=None
):
    """Extract the rule which leas to a particular prediction.


    Arguments:
    * dt - decision tree from which the decision rule needs to be extracted.
    * x - instance from which to extract the decision rule.
    * features_names

    Returns:
    * dr - decision rule as a str of the form '(X[0] > 2.0) AND (X[4] > -2.0)'
    """

    rules = []

    if isinstance(dt, DecisionTreeClassifier):
        dt = dt.tree_

    if not features_names:
        features_names = [f"X[{i}]" for i in range(dt.feature.shape[0])]
    else:
        assert len(features_names) == dt.feature.shape[0]

    threshold_values = dt.threshold

    def traverse(node: int, rule: str):
        """
        Recursively traverse dt and build up the rule.
        """
        feat_node = dt.feature[node]

        if feat_node == _tree.TREE_UNDEFINED:
            # append the string into rules so it can be returned.
            rules.append(rule)
            return

        feat_name = features_names[feat_node]
        thrs_val = threshold_values[feat_node]

        if x[features_names.index(feat_name)] <= thrs_val:
            traverse(dt.children_left[node], rule + f"({feat_name} <= {thrs_val}) AND ")
        else:
            traverse(dt.children_right[node], rule + f"({feat_name} > {thrs_val}) AND ")

    traverse(0, "")

    return rules[0][:-5]


def counterfactuals():
    pass
