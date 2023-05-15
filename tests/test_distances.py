from pylore.distances import LOREDistance
import numpy as np


def test_euclidean_distance():
    dist = LOREDistance().euclidean_distance

    a = np.random.random(100)
    b = np.random.random(100)
    assert dist(a, a) == 0
    assert np.abs(dist(a, b) - dist(b, a)) <= 1e15
    assert dist(a, b) > 0 and dist(a, -b) > 0

    x = np.array([1, 0, -5])
    y = np.array([-3, 2, -1])

    assert dist(x, y) == 6
