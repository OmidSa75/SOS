import numpy as np


class SOS:
    GOODS_THRESHOLD = 0.5

    def __init__(
            self,
            l_bound,
            u_bound,
            population_size,
            fitness_vector_size,
            environment,
            individual_class,
    ):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population_size = population_size
        self.fitness_vector_size = fitness_vector_size
        self.population = None
        self.good_population = None
        self.bad_population = None
        self.best = None
        self.environment = environment
        self.create_individual = individual_class

        self.goods_threshold = int(self.population_size * self.GOODS_THRESHOLD)

    def float_rand(self, lower_bound, upper_bound, size=None):
        return lower_bound + ((upper_bound - lower_bound) * np.random.random(size))

    def generate_population(self):
        population = self.float_rand(self.l_bound, self.u_bound, (self.population_size, self.fitness_vector_size))
        self.population = [self.create_individual(p, self.environment) for p in population]
        # self.best = sorted(self.population, key=lambda x: x.fitness_value)[0]

    def mutualism(self, a_index, population):
        b_index = np.random.choice(np.delete(np.arange(len(population)), a_index))
        b = population[b_index]
        a = population[a_index]
        bf1, bf2 = np.random.randint(0, 3, 2)  # benefit factor1
        array_rand = np.random.random(self.fitness_vector_size)
        mutual = (a.phenotypes + b.phenotypes) / 2
        new_a = a.phenotypes + (array_rand * (self.best.phenotypes - (mutual * bf1)))
        new_b = b.phenotypes + (array_rand * (self.best.phenotypes - (mutual * bf2)))
        new_a = self.create_individual(np.clip(new_a, a_min=self.l_bound, a_max=self.u_bound),
                                       self.environment)
        new_b = self.create_individual(np.clip(new_b, a_min=self.l_bound, a_max=self.u_bound),
                                       self.environment)
        population[a_index] = new_a if (new_a.fitness_value < a.fitness_value and new_a.constraints[0] == 0) else a
        population[b_index] = new_b if (new_b.fitness_value < b.fitness_value and new_b.constraints[0] == 0) else b
        return population

    def commensalism(self, a_index, population):
        b_index = np.random.choice(np.delete(np.arange(len(population)), a_index))
        b = population[b_index]
        a = population[a_index]
        array_rand = self.float_rand(self.l_bound, self.u_bound, self.fitness_vector_size)
        new_a = a.phenotypes + (array_rand * (self.best.phenotypes - b.phenotypes))
        new_a = self.create_individual(np.clip(new_a, a_min=self.l_bound, a_max=self.u_bound), self.environment)
        population[a_index] = new_a if (new_a.fitness_value <= a.fitness_value and new_a.constraints[0] == 0) else a
        return population

    def parasitism(self, a_index, population):
        parasite = np.array(population[a_index].phenotypes)
        b_index = np.random.choice(np.delete(np.arange(len(population)), a_index))
        b = population[b_index]
        index = np.random.randint(0, self.fitness_vector_size)
        parasite[index] = self.float_rand(self.l_bound, self.u_bound)[index]
        parasite = self.create_individual(parasite, self.environment)
        population[b_index] = parasite if (parasite.fitness_value <= b.fitness_value and parasite.constraints[0] == 0) else b
        return population

    def evaluate_population(self):
        good_indices = tuple(True if ind.constraints[0] == 0 else False for ind in self.population)
        goods = []
        bads = []
        for i, flag in enumerate(good_indices):
            if flag:
                goods.append(self.population[i])
            else:
                bads.append(self.population[i])
        violated_con = np.asarray([ind.constraints[1:] for ind in bads])
        z_scores = (violated_con - np.mean(violated_con, axis=0)) / np.std(violated_con, axis=0)
        z_scores[np.isnan(z_scores)] = 0
        scores = np.sum(z_scores, axis=1)
        self.bad_population = np.asarray(bads)[np.argsort(scores)].tolist()
        self.good_population = goods
        # todo: handle zero good values
        self.best = sorted(self.good_population, key=lambda x: x.fitness_value)[0]
        return goods, bads

    def proceed(self, steps):
        for j in range(1, steps + 1):
            self.evaluate_population()
            # print(len(self.good_population))
            for i, _ in enumerate(self.good_population):
                self.mutualism(i, self.good_population)
                self.commensalism(i, self.good_population)
                self.parasitism(i, self.good_population)

            if len(self.bad_population) > 2:
                for i, _ in enumerate(self.bad_population):
                    self.parasitism(i, self.bad_population)

            if j % 5 == 0:
                print('{0}/{1} Current population:'.format(j, steps))
                print(self.best)
            self.population = self.good_population + self.bad_population
