import pytest
from pylore.genetics import KPointCrossoverer
import numpy as np


@pytest.fixture
def data():
    return np.random.randn(100, 20)


def test_kpoint_crossover(data):
    K = 2
    crossover = KPointCrossoverer(0.5, K)
    new_data = crossover(data)
    assert len(new_data) <= len(data) / 2
