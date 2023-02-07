import numpy as np


class SOS:
    def __init__(
            self,
            l_bound,
            u_bound,
            population_size,
            fitness_vector_size,
            fitness_function,
            individual_class,
    ):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population_size = population_size
        self.fitness_vector_size = fitness_vector_size
        self.population = None
        self.best = None
        self.fitness_function = fitness_function
        self.create_individual = individual_class

    def float_rand(self, a, b, size=None):
        return a + ((b - a) * np.random.random(size))

    def generate_population(self):
        population = self.float_rand(self.l_bound, self.u_bound, (self.population_size, self.fitness_vector_size))
        self.population = [self.create_individual(p, self.fitness_function) for p in population]
        self.best = sorted(self.population, key=lambda x: x.fitness_value)[0]

    def mutualism(self, a_index):
        b_index = np.random.choice(np.delete(np.arange(self.population_size), a_index))
        b = self.population[b_index]
        a = self.population[a_index]
        bf1, bf2 = np.random.randint(1, 3, 2)  # benefit factor1
        array_rand = np.random.random(self.fitness_vector_size)
        mutual = (a.phenotypes + b.phenotypes) / 2
        new_a = a.phenotypes + (array_rand * (self.best.phenotypes - (mutual * bf1)))
        new_b = b.phenotypes + (array_rand * (self.best.phenotypes - (mutual * bf2)))
        new_a = self.create_individual([self.u_bound if x > self.u_bound
                                        else self.l_bound if x < self.l_bound else x for x in new_a],
                                       self.fitness_function)
        new_b = self.create_individual([self.u_bound if x > self.u_bound
                                        else self.l_bound if x < self.l_bound else x for x in new_b],
                                       self.fitness_function)
        self.population[a_index] = new_a if new_a.fitness_value < a.fitness_value else a
        self.population[b_index] = new_b if new_b.fitness_value < b.fitness_value else b

    def commensalism(self, a_index):
        b_index = np.random.choice(np.delete(np.arange(self.population_size), a_index))
        b = self.population[b_index]
        a = self.population[a_index]
        array_rand = self.float_rand(self.l_bound, self.u_bound, self.fitness_vector_size)
        new_a = a.phenotypes + (array_rand * (self.best.phenotypes - b.phenotypes))
        new_a = self.create_individual([self.u_bound if x > self.u_bound
                                        else self.l_bound if x < self.l_bound
                                        else x for x in new_a], self.fitness_function)
        self.population[a_index] = new_a if new_a.fitness_value <= a.fitness_value else a

    def parasitism(self, a_index):
        parasite = np.array(self.population[a_index].phenotypes)
        b_index = np.random.choice(np.delete(np.arange(self.population_size), a_index))
        b = self.population[b_index]
        parasite[np.random.randint(0, self.fitness_vector_size)] = self.float_rand(self.l_bound, self.u_bound)
        parasite = self.create_individual(parasite, self.fitness_function)
        self.population[b_index] = parasite if parasite.fitness_value <= b.fitness_value else b

    def proceed(self, steps):
        for j in range(1, steps + 1):
            for i, val in enumerate(self.population):
                self.mutualism(i)
                self.commensalism(i)
                self.parasitism(i)
                self.best = sorted(self.population, key=lambda x: x.fitness_value)[0]
            if j % 50 == 0:
                print('{0}/{1} Current population:'.format(j, steps))
                print(self.best)
