import pytest
from pylore.treeutils import extract_decision_rule
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import re


@pytest.fixture
def dataset():
    X = np.random.randint(0, 10, size=(20, 10))
    y = np.random.randint(5, size=(20,))
    return X, y


@pytest.fixture
def decision_tree_classifier(dataset):
    X, y = dataset
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    return dt


def test_decision_rule_format(decision_tree_classifier, dataset):
    X, _ = dataset
    dr = extract_decision_rule(decision_tree_classifier, X[0])
    assert isinstance(dr, str)
    assert " AND " in dr

    def check_parentheses(lst):
        pattern = r"^\(.*\)$"
        for element in lst:
            assert re.match(pattern, element)

    check_parentheses(dr.split(" AND "))
