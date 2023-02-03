import numpy as np
from .base_env import IBaseEnv


class PressureVessel(IBaseEnv):
    THICKNESS_STEP = 0.0625

    def fitness_function(self, individual: np.ndarray):
        x1 = np.round(individual[:, 0] / self.THICKNESS_STEP, 0) * self.THICKNESS_STEP
        x2 = np.round(individual[:, 1] / self.THICKNESS_STEP, 0) * self.THICKNESS_STEP
        x3 = individual[:, 2]
        x4 = individual[:, 3]
        return self.THICKNESS_STEP * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3

    def constraints(self, individual):
        gs = self.g(individual)
        res = []
        # loop over each individual
        for ind in gs:

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

            res.append([count, g1, g2, g3, g4])

        return np.asarray(res)

    def g(self, individual):
        x1 = individual[:, 0]
        x2 = individual[:, 1]
        x3 = individual[:, 2]
        x4 = individual[:, 3]

        g1 = - x1 + 0.0193 * x3
        g2 = - x2 + 0.00954 * x3
        g3 = - np.pi * x3**2 * x4 - (4 * np.pi / 3) * x3**3 + 1296000
        g4 = x4 - 240
        return np.array([g1, g2, g3, g4]).T


