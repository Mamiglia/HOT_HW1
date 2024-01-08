import numpy as np
from .Solution import Solution
from funcs import VariableNeighborhoodDescent, Flip1, QualityFlip1, SwapNode
from itertools import chain

from funcs.Greedy import Karger
from funcs.Greedy import deletion_heuristic
from funcs.Greedy import GreedySPlex
from .VND import partial_local_search
class GeneticAlgorithm_modified:
    def __init__(self, A, W, S, length_population: int = 100, mut: bool=True, k: float=0.0) -> None:
        self.A = A
        self.W = W
        self.S = S
        self.length_population = length_population
        self.mut= mut
        self.k = k

        # Initialize population
        len_karger = int(length_population*0.75)
        len_random = length_population-len_karger-1

        greedy_karger = Karger(A, W, S)
        def karger_start():
            A1, splexes = greedy_karger.random_solution()
            return Solution.build(A, W, A1, splexes)
        self.population = [karger_start() for _ in range(len_karger)]

        greedy_random = GreedySPlex(A, W, S)
        def random_start():
            A1, splexes = greedy_random.random_solution()
            return Solution.build(A, W, A1, splexes)
        self.population.extend([random_start() for _ in range(len_random)])

        greedy_optimal = GreedySPlex(A, W, S)
        def opt_start():
            A1, splexes = greedy_optimal.solution()
            return Solution.build(A, W, A1, splexes)
        self.population.append(opt_start())
    
    def get_population(self) -> list:
        return self.population

    def get_fitness(self) -> list:
        return [solution.obj() for solution in self.population]

    def next_generation(self, population: list) -> list:        
        parent1 = self.selection(population)
        parent2 = self.selection(population)
        child = self.crossover(parent1, parent2)
        child = self.mutation(child)
        population = self.survival(population, child)

        return population
    
    def selection(self, population: list, random: bool=True) -> Solution:
        if random:
            return population[np.random.randint(len(population))]
        else:
            prob = np.array([solution.obj() for solution in population])/sum([solution.obj() for solution in population])
            return np.random.choice(population, p=prob)
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        child_W = parent1.W
        child_A1 = parent1.A1 * parent2.A1
        child_clusters = [[i] for i in range(child_W.shape[0])]
        to_remove = []
        for i in range(child_W.shape[0]):
            for j in range(i, child_W.shape[0]):
                if child_A1[i][j] == 1:
                    child_clusters[i].append(j)
                    to_remove.append(j)

        child_clusters = [c for i,c in enumerate(child_clusters) if i not in to_remove]

        P = np.random.rand()
        for i in range(10):
            if self.k <= P:
                lengths = [len(c) for c in child_clusters]
                max_len = max(lengths)
                prob_cl = np.array([max_len - l for l in lengths])/sum([max_len - l for l in lengths])
                i = np.random.choice(len(child_clusters), p=prob_cl)
                j = np.random.choice(len(child_clusters), p=prob_cl)
                if i != j:
                    child_clusters[i].extend(child_clusters[j])
                    child_clusters.pop(j)

        child_A1 = sum(deletion_heuristic(child_W, plex=cl, s=self.S) for cl in child_clusters)
        child_splexes = {i:plex for plex in child_clusters for i in plex}
        child = Solution.build(self.A, child_W, child_A1, child_splexes)

        return child

    
    def mutation(self, child: Solution) -> Solution:
        try:
            if self.mut:
                neighbourhood = [SwapNode(self.A.shape[0])]
                local_search = partial_local_search(neighbourhood, 1)
                child = local_search.search(child)
                neighbourhood = [Flip1(self.S)]
                local_search = partial_local_search(neighbourhood, 10)
                child = local_search.search(child)
        finally:
            return child
    
    def survival(self, population: list, child: Solution) -> list:
        worst_fitness = child.obj()
        idx = -1
        for i, solution in enumerate(population):
            if solution.obj() > worst_fitness:
                worst_fitness = solution.obj()
                idx = i
        
        if idx != -1:
            population[idx] = child
            # print('Replaced')

        return population
    
    def evolution(self, max_generations: int = 1000) -> Solution:
        population = self.population
        for generation in range(max_generations):
            # print(f'Generation {generation}')
            population = self.next_generation(population)
            if generation % 100 == 0:
                print(f'Generation {generation}')
                print(f'Mean fitness: {np.mean([solution.obj() for solution in population])}')
                print(f'Best fitness: {min([solution.obj() for solution in population])}')

        return population

