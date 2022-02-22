from pylore.distances import EuclideanDistance
import numpy as np


def test_euclidean_distance():
    dist = EuclideanDistance()

    a = np.random.random(100)
    b = np.random.random(100)
    assert dist(a, a) == 0
    assert dist(a, b) == dist(b, a)
    assert dist(a, b) > 0
    assert dist(a, -b) > 0

    x = np.array([1, 0, -5])
    y = np.array([-3, 2, -1])

    assert dist(x, y) == 6
