"""
This module contains utilities for extracting decision rules
and counterfactuals from a DecisionTreeClassifier.
"""
from sklearn.tree import DecisionTreeClassifier, _tree
from typing import Union


def extract_decision_rule(
    dt: Union[DecisionTreeClassifier, _tree.Tree], x, features_names=None
) -> str:
    """Extract the rule which lead to a particular prediction.


    Arguments:
    * dt - decision tree from which the decision rule needs to be extracted.
    * x - instance from which to extract the decision rule.
    * features_names - list containing the name of the features.

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

    def traverse(node: int, rule: str):
        """
        Recursively traverse dt and build up the rule.
        """
        feat_node = dt.feature[node]

        if feat_node == _tree.TREE_UNDEFINED:
            rules.append(rule[:-5])  # remove the last AND
            return

        feat_name = features_names[feat_node]
        threshold_val = dt.threshold[feat_node]

        if x[features_names.index(feat_name)] <= threshold_val:
            traverse(
                dt.children_left[node],
                rule + f"({feat_name} <= {threshold_val}) AND ",
            )
        else:
            traverse(
                dt.children_right[node],
                rule + f"({feat_name} > {threshold_val}) AND ",
            )

    traverse(0, "")

    return rules[0]


def extract_counterfactuals(
    dt: Union[DecisionTreeClassifier, _tree.Tree],
    rule: Union[str, list[str]],
    x,
):
    """ """
    if isinstance(dt, DecisionTreeClassifier):
        dt = dt.tree_

    if isinstance(rule, str):
        rule = rule.split("  AND  ")
