from pylore.lore import indicator, fitness_equal, fitness_nequal
import numpy as np


def test_indicator():
    assert indicator(True) == 1
    assert indicator(False) == 0


def test_fitnesses():
    pass
