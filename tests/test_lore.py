import pytest
from pylore import LORE, LOREDistance
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def dummy_dataset():
    return []


@pytest.fixture
def dummy_model():
    return LogisticRegression()


@pytest.fixture
def dummy_distance():
    return LOREDistance(categorical_mask=[])


def test_instantiation(dummy_model, dummy_distance):
    lore = LORE(dummy_model, 100, dummy_distance)
    assert lore
