import pytest
from pylore.treeutils import extract_decision_rule
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import re


@pytest.fixture
def dataset():
    X, y = make_classification(
        n_samples=200, n_features=15, n_informative=5, n_classes=4
    )
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
    assert " AND " in dr or len(dr.split(" AND ")) == 1

    def check_parentheses(lst):
        pattern = r"^\(.*\)$"
        for element in lst:
            assert re.match(pattern, element)

    check_parentheses(dr.split(" AND "))
