from funcs import readin, Solution, GA, is_splex
from funcs.Greedy import Karger
import numpy as np

# file = 'data/test_instances/heur030_n_330_m_5613.txt'
# file = 'data/test_instances/heur022_n_322_m_14226.txt'
# file = 'data/test_instances/heur039_n_361_m_13593.txt'


# file = 'data\inst_competition\heur049_n_300_m_17695.txt'
file = 'data\inst_competition\heur050_n_300_m_19207.txt'
# file = 'data\inst_competition\heur051_n_300_m_20122.txt'

S,A,W = readin(file)
greedy = Karger(A, W, S)
def random_start():
    A1, splexes = greedy.random_solution()
    return Solution.build(A,W,A1, splexes)


# Genetic Algorithm
length_population = 100
initial_population = [random_start() for _ in range(length_population)]
GeneticAlgorithm = GA.GeneticAlgorithm(initial_population, A, S)
# parent1 = GeneticAlgorithm.selection(initial_population)
# parent2 = GeneticAlgorithm.selection(initial_population)
# child = GeneticAlgorithm.crossover(parent1, parent2)
# if is_splex(child.A1, S):
#     print('Yes')

print(np.mean(GeneticAlgorithm.get_fitness()))
final_population = GeneticAlgorithm.evolution(3000)
print(np.min([solution.obj() for solution in final_population]))

# for i in range(len(final_population)):
#     if  not is_splex(final_population[i].A1, S):
#         print('Attention: There\'s an Error!!!')
#         break

