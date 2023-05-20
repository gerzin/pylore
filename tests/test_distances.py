from pylore.distances import LOREDistance
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mixed_random_dataset():
    df = pd.DataFrame()
    df["A"] = np.random.randint(0, 10, size=(10,))
    df["B"] = "Ciao"
    df["C"] = np.random.randint(0, 10, (10,))
    df["D"] = "Cane"
    return df


def test_euclidean_distance():
    dist = LOREDistance.normalized_eucliden

    a = np.random.random(100)
    b = np.random.random(100)
    assert dist(a, a) == 0
    assert np.abs(dist(a, b) - dist(b, a)) <= 1e15
    assert dist(a, b) >= 0 and dist(a, -b) >= 0

    x = np.array([1, 0, -5])
    y = np.array([-3, 2, -1])

    assert dist(x, y) == 6


def test_simplematch_distance():
    dist = LOREDistance.simple_match

    a = ""
    b = ""
    assert dist(a, a) == 0
    assert np.abs(dist(a, b) - dist(b, a)) == 0

    c = np.array([*"ciao"])
    d = np.array([*"cane"])
    assert dist(c, d) == 3
    assert dist(c, c) == 0
    assert np.abs(dist(c, d) - dist(d, c)) == 0

    e, f = np.array([*"abc"]), np.array([*"def"])
    assert dist(e, f) == len(e)


def test_combined_distance(mixed_random_dataset):
    dist = LOREDistance(mixed_random_dataset)

    assert len(mixed_random_dataset) > 3
    x = mixed_random_dataset.iloc[0].to_numpy()
    y = mixed_random_dataset.iloc[1].to_numpy()
    if not (x == y).all():
        assert dist(x, y) > 0
    else:
        assert dist(x, y) == 0
