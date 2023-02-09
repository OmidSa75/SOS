import numpy as np

from sos.sos import SOS
from environment.individual import Individual
from environment.pressure_vessel import PressureVessel


def test_bounds():
    upper_bound = np.array([6.1875, 6.1875, 100, 200])
    lower_bound = np.array([0.0625, 0.0625, 10, 10])
    dim = 4
    population_size = 100
    sos = SOS(lower_bound, upper_bound, population_size, dim, PressureVessel, Individual)
    sos.generate_population()
    for ind in sos.population:
        assert (ind <= upper_bound).all()
        assert (ind >= lower_bound).all()


def test_create_population():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel, Individual)
    sos.generate_population()
    assert len(sos.population) == population_size
    assert sos.population[0].phenotypes.shape == (dim,)
    assert isinstance(sos.population[0].fitness_value, float)


def test_mutualism():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel, Individual)
    sos.generate_population()

    population_index = 5
    sos.mutualism(5)


def test_commensalism():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel, Individual)
    sos.generate_population()

    population_index = 5
    sos.commensalism(5)


def test_parasitism():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel, Individual)
    sos.generate_population()

    population_index = 5
    sos.parasitism(5)
