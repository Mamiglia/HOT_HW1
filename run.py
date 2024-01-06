from funcs import readin, Solution, GA, is_splex
from funcs.Greedy import Karger
import numpy as np

# file = 'data/test_instances/heur001_n_10_m_31.txt'
file = 'data/test_instances/heur011_n_250_m_6574.txt'

S,A,W = readin(file)
greedy = Karger(A, W, S)
def random_start():
    A1, splexes = greedy.random_solution()
    return Solution.build(A,W,A1, splexes)


# Genetic Algorithm
length_population = 10
initial_population = [random_start() for _ in range(length_population)]
GeneticAlgorithm = GA.GeneticAlgorithm(initial_population, A, S)

print(min(GeneticAlgorithm.get_fitness()))
final_population = GeneticAlgorithm.evolution(10)
print(min([solution.obj() for solution in final_population]))

for i in range(len(final_population)):
    if  not is_splex(final_population[i].A1, S):
        print('Attention: There\'s an Error!!!')
        break

