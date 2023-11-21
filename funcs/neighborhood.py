from typing import Iterator, List
from abc import abstractmethod, ABCMeta
import numpy as np
from funcs.Solution import Solution

class Neighborhood(metaclass=ABCMeta):
    @abstractmethod
    def shaking(self, x : Solution) -> Solution:
        pass

    @abstractmethod
    def neighbors(self, x : Solution) -> Iterator[Solution]:
        pass

    def neighbor_list(self, x: Solution) -> List[Solution]:
        return list(self.neighbors(x))


class Flip1(Neighborhood):
    def neighbors(self, x: Solution) -> Iterator[Solution]:
        for i in range(x.size):
            for j in range(i):
                x_n = self.step(x, i, j)
                yield x_n
    
    def shaking(self, x : Solution) -> Solution:
        i = np.random.randint(x.size)
        j = np.random.randint(x.size)
        return self.step(x, i, j)
        

    def step(self, x : Solution, i: int, j: int) -> Solution:
        x_n = x.copy()
        x_n.A1[i,j] = 1 - x_n.A1[i,j]
        x_n.A1[j,i] = x_n.A1[i,j] 

        # Delta evaluation
        x_n._obj = x.obj() + ((x_n.A[i,j] != x_n.A1[i,j])*2-1)*x.W[i,j]
        return x_n


