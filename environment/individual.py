import numpy as np


class Individual:
    def __init__(self, phenotypes, environment):
        self.phenotypes = np.array(phenotypes)
        assert hasattr(environment, 'fitness_function')
        self.fitness_value = environment.fitness_function(self.phenotypes)
        self.constraints = environment.constraints(self.phenotypes) if hasattr(environment, 'constraints') else None

    def __str__(self):
        return f"{self.phenotypes} = {self.fitness_value}"

    def __repr__(self):
        return self.__str__()

    def __ge__(self, other):
        return self.phenotypes >= other

    def __gt__(self, other):
        return self.phenotypes > other

    def __le__(self, other):
        return self.phenotypes <= other

    def __lt__(self, other):
        return self.phenotypes < other

    def __eq__(self, other):
        return self.phenotypes == other
