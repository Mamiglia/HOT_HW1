import numpy as np
from .Solution import Solution
from funcs import VariableNeighborhoodDescent, SwapNode

def deletion_heuristic(A, plex, s):
    '''given some plex nodes remove all the possible edges starting from the biggest one until possible'''
    if len(plex) == 1:
        return np.zeros_like(A, dtype=np.int32)
    cluster_idx = np.ix_(plex, plex)
    Ac = A[cluster_idx]     # Adjacency matrix for the cluster only
    min_edges = len(plex) - s
    sorted_edges = np.argsort(Ac, axis=-1)[:,s::-1]
    A1 = np.ones_like(Ac, dtype=np.int32)
    np.fill_diagonal(A1, 0)
    
    for i in range(Ac.shape[0]):
        extra_edges = min(A1[i].sum() - min_edges, A1.shape[1])
        for k in range(extra_edges):
            to_remove = sorted_edges[i,k]
            if Ac[i,to_remove] < 0:
                break
            if A1[i,to_remove] == 0 or A1[to_remove].sum() <= min_edges:
                continue
            A1[i,to_remove] = 0
            A1[to_remove,i] = 0
    res = np.zeros_like(A, dtype=np.int32)
    res[cluster_idx] = A1
    return res

class GeneticAlgorithm:
    def __init__(self, population: list, A, S) -> None:
        self.population = population
        self.A = A
        self.S = S 
    
    def get_population(self) -> list:
        return self.population

    def get_fitness(self) -> list:
        return [solution.obj() for solution in self.population]

    def next_generation(self, population: list) -> list:        
        parent1 = self.selection(population)
        parent2 = self.selection(population)
        child = self.crossover(parent1, parent2)
        # child = self.mutation(child)
        population = self.survival(population, child)

        return population
    
    def selection(self, population: list) -> Solution:
        return population[np.random.randint(len(population))]
    
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
        child_A1 = sum(deletion_heuristic(child_W, plex=cl, s=self.S) for cl in child_clusters)
        child_splexes = {i:plex for plex in child_clusters for i in plex}
        child = Solution.build(self.A, child_W, child_A1, child_splexes)
        local_search = VariableNeighborhoodDescent([SwapNode(self.A.shape[0])])
        child = local_search.search(child)
        return child
    
    def mutation(self, child: Solution) -> Solution:
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

        return population
    
    def evolution(self, max_generations: int = 1000) -> Solution:
        population = self.population
        for generation in range(max_generations):
            population = self.next_generation(population)

        return population
    