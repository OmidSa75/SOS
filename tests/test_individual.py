import numpy as np

from environment.individual import Individual
from environment.pressure_vessel import PressureVessel


def test_individual():
    individual = Individual(np.array([1, 0, -1, 3]), PressureVessel)
    flags = individual > np.array([0, 0, 0, 0])
    assert flags.sum() == 2
    flags = individual >= np.array([0, 0, 0, 0])
    assert flags.sum() == 3
    flags = individual < np.array([0, 0, 0, 0])
    assert flags.sum() == 1
    flags = individual <= np.array([0, 0, 0, 0])
    assert flags.sum() == 2
    flags = individual == np.array([0, 0, 0, 0])
    assert flags.sum() == 1
