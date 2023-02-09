import numpy as np
from sos.sos import SOS
from environment.individual import Individual
from environment.pressure_vessel import PressureVessel
from eval.plots import plot_boxplot, plot_values
from eval.statistics import get_statistical_info


if __name__ == '__main__':
    upper_bound = np.array([6.1875, 6.1875, 100, 200])
    lower_bound = np.array([0.0625, 0.0625, 10, 10])
    population_size = 100
    steps = 200
    dim = 4

    best_fitness = []
    for _ in range(100):
        sos = SOS(
            lower_bound,
            upper_bound,
            population_size,
            dim,
            PressureVessel,
            Individual,
        )
        sos.generate_population()
        sos.proceed(steps)
        # print(sos.best)
        # print(sos.best.constraints)
        # plot_values(steps, sos.q_mean_fitness, 'mean fitness')
        # plot_values(steps, sos.q_best_fitness, 'best fitness')
        # print(get_statistical_info(sos.q_best_fitness))
        best_fitness.append(sos.best.fitness_value)
    plot_boxplot(best_fitness, 'best fitness boxplot')
