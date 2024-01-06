import os
from funcs import readin, Solution, GA
from funcs.Greedy import Karger


file = 'data/test_instances/heur001_n_10_m_31.txt'

S,A,W = readin(file)
greedy = Karger(A, W, S)
def random_start():
    A1, splexes = greedy.random_solution()
    # print('Ran greedy')
    return Solution.build(A,W,A1, splexes)


# Genetic Algorithm
length_population = 10
initial_population = [random_start() for _ in range(length_population)]
GeneticAlgorithm = GA.GeneticAlgorithm(initial_population)
print(GeneticAlgorithm.get_fitness())
