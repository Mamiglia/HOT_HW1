import dataclasses as dc
from typing import Collection, Iterator, List
from abc import abstractmethod, ABCMeta
import numpy as np
import numpy.typing as npt

@dc.dataclass
class Solution(frozen=True):
    A: npt.NDArray[np.int_]
    W: npt.NDArray[np.int_]
    A1: npt.NDArray[np.int_]

    def X(self) -> npt.NDArray[np.bool_]:
        return self.A != self.A1
    
    def n(self) -> int:
        return self.A.shape[0]
    
    def copy(self):
        return Solution(self.A, self.W, self.A1.copy())

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
        for i in range(x.n()):
            for j in range(i):
                x_n = self.step(x, i, j)
                yield x_n
    
    def shaking(self, x : Solution) -> Solution:
        i = np.random.randint(x.n())
        j = np.random.randint(x.n())
        return self.step(x, i, j)
        

    def step(self, x : Solution, i: int, j: int) -> Solution:
        x_n = x.copy()
        x_n.A1[i,j] = 1 - x_n.A1[i,j]
        x_n.A1[j,i] = 1 - x_n.A1[j,i]
        return x_n


