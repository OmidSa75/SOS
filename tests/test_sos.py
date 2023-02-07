from sos.sos import SOS
from environment.individual import Individual
from environment.pressure_vessel import PressureVessel


def test_create_population():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel.fitness_function, Individual)
    sos.generate_population()
    assert len(sos.population) == population_size
    assert sos.population[0].phenotypes.shape == (dim,)
    assert isinstance(sos.population[0].fitness_value, float)


def test_mutualism():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel.fitness_function, Individual)
    sos.generate_population()

    population_index = 5
    sos.mutualism(5)


def test_commensalism():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel.fitness_function, Individual)
    sos.generate_population()

    population_index = 5
    sos.commensalism(5)


def test_parasitism():
    interval = (-1, 1)
    population_size = 100
    dim = 4
    sos = SOS(interval[0], interval[1], population_size, dim, PressureVessel.fitness_function, Individual)
    sos.generate_population()

    population_index = 5
    sos.parasitism(5)
