from pylore.distances import EuclideanDistance, dist_from_str
import numpy as np
import pytest


def test_dist_from_str():
    d = dist_from_str("euclidean")()
    assert isinstance(d, EuclideanDistance)


def test_dist_from_str_error():
    with pytest.raises(KeyError):
        _ = dist_from_str("jibberishasdjkl")


def test_euclidean_distance():
    dist = EuclideanDistance()

    a = np.random.random(100)
    b = np.random.random(100)
    assert dist(a, a) == 0
    assert np.abs(dist(a, b) - dist(b, a)) <= 1e15
    assert dist(a, b) > 0 and dist(a, -b) > 0

    x = np.array([1, 0, -5])
    y = np.array([-3, 2, -1])

    assert dist(x, y) == 6
