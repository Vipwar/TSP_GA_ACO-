import random
from utils.tsp_utils import total_distance

class GA:
    def __init__(self, cities, pop_size=50, generations=200, mutation_rate=0.1):
        self.cities = cities
        self.n = len(cities)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.history = []

    def _create_individual(self):
        path = list(range(self.n))
        random.shuffle(path)
        return path

    def _crossover(self, p1, p2):
        a, b = sorted(random.sample(range(self.n), 2))
        child = [-1] * self.n
        child[a:b] = p1[a:b]

        ptr = 0
        for x in p2:
            if x not in child:
                while child[ptr] != -1:
                    ptr += 1
                child[ptr] = x
        return child

    def _mutate(self, ind):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n), 2)
            ind[i], ind[j] = ind[j], ind[i]

    def run_stepwise(self):
        pop = [self._create_individual() for _ in range(self.pop_size)]
        best = min(pop, key=lambda x: total_distance(x, self.cities))

        for gen in range(self.generations):
            new_pop = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(pop, 2)
                child = self._crossover(p1, p2)
                self._mutate(child)
                new_pop.append(child)

            pop = sorted(new_pop, key=lambda x: total_distance(x, self.cities))
            if total_distance(pop[0], self.cities) < total_distance(best, self.cities):
                best = pop[0]

            best_dist = total_distance(best, self.cities)
            self.history.append((gen, best_dist))
            yield gen, best, best_dist
