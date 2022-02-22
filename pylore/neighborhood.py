from pylore.blackbox import AbstractBlackBoxWrapper
from typing import Callable


def genetic_neighborhood(x, fitness_fn: Callable, b: AbstractBlackBoxWrapper, n: int, g: int, pc: float, pm: float):
    """

    :param x: instance to explain.
    :param fitness_fn: fitness function.
    :param b: black box.
    :param n: population size.
    :param g: number of generations.
    :param pc: probability of crossover.
    :param pm: probability of mutation.
    :return:
    """
    print("Hello from generic")