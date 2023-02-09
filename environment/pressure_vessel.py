import numpy as np
from .base_env import IBaseEnv
from .individual import Individual


class PressureVessel(IBaseEnv):
    THICKNESS_STEP = 0.0625

    @classmethod
    def fitness_function(cls, individual: np.ndarray):
        x1 = np.round(individual[0] / cls.THICKNESS_STEP, 0) * cls.THICKNESS_STEP
        x2 = np.round(individual[1] / cls.THICKNESS_STEP, 0) * cls.THICKNESS_STEP
        x3 = individual[2]
        x4 = individual[3]
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 * x3 + 3.1661 * x1 * x1 * x4 + 19.84 * x1 * x1 * x3

    @classmethod
    def constraints(cls, individual):
        ind = cls.g(individual)

        count = 0  # number of times ind broke a constraint
        if ind[0] > 0:
            g1 = ind[0]  # broke constraint difference
            count += 1
        else:
            g1 = 0

        if ind[1] > 0:
            g2 = ind[1]  # broke constraint difference
            count += 1
        else:
            g2 = 0

        if ind[2] > 0:
            g3 = ind[2]  # broke constraint difference
            count += 1
        else:
            g3 = 0

        if ind[3] > 0:
            g4 = ind[3]  # broke constraint difference
            count += 1
        else:
            g4 = 0

        return np.asarray([count, g1, g2, g3, g4])

    @classmethod
    def g(cls, individual):
        x1 = individual[0]
        x2 = individual[1]
        x3 = individual[2]
        x4 = individual[3]

        g1 = - x1 + 0.0193 * x3
        g2 = - x2 + 0.00954 * x3
        g3 = - np.pi * x3 ** 2 * x4 - (4 * np.pi / 3) * x3 ** 3 + 1296000
        g4 = x4 - 240
        return np.array([g1, g2, g3, g4])
