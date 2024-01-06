import numpy as np
from .Solution import Solution

class GeneticAlgorithm:
    def __init__(self, population: list) -> None:
        self.population = population

    def get_fitness(self) -> list:
        return [solution.obj() for solution in self.population]

    def next_generation(self, population: list) -> list:        
        parent1 = self.selection(population)
        parent2 = self.selection(population)
        child = self.crossover(parent1, parent2)
        child = self.mutation(child)
        population = self.survival(population, child)

        return population
    
    def selection(self, population: list) -> Solution:
        return population[np.random.randint(len(population))]
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        return parent1
    
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
