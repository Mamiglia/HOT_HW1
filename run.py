from funcs import readin, Solution, GA
from funcs.Greedy import Karger
import numpy as np

file = 'data/test_instances/heur001_n_10_m_31.txt'
# file = 'data/test_instances/heur011_n_250_m_6574.txt'

S,A,W = readin(file)
greedy = Karger(A, W, S)
def random_start():
    A1, splexes = greedy.random_solution()
    return Solution.build(A,W,A1, splexes)


# Genetic Algorithm
length_population = 10
initial_population = [random_start() for _ in range(length_population)]
GeneticAlgorithm = GA.GeneticAlgorithm(initial_population, A)
parent1 = initial_population[0]
parent2 = initial_population[1]
# child = GeneticAlgorithm.crossover(parent1, parent2)


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
child_A1 = sum(deletion_heuristic(child_W, plex=cl, s=S) for cl in child_clusters)
child_splexes = {i:plex for plex in child_clusters for i in plex}
child = Solution.build(A, child_W, child_A1, child_splexes)

print(child.obj())
print(parent1.obj())
print(parent2.obj())



# child_clusters = [[i] for i in range(child_W.shape[0])]
# for i in range(child_W.shape[0]):
#     for j in range(child_W.shape[0]):
#         if child_A1[i][j] == 1:
#             child_clusters[i].append(j)
# child_splexes = {i:plex for plex in child_clusters for i in plex}


# A1 = sum(deletion_heuristic(child_W, plex=cl, s=self.s) for cl in clusters)
# child = Solution.build(A, child_W, child_A1, child_splexes)

# common_el = list(set(parent1.get_cluster()[0]).intersection(set(parent2.get_cluster()[0])))
# print(common_el)
# print(child.get_cluster()[0])
