import pytest
from pylore import LORE
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def dummy_model():
    return LogisticRegression()


def test_instantiation(dummy_model):
    lore = LORE(dummy_model, 100)
    assert lore
