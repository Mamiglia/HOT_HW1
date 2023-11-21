import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood
from .VND import VariableNeighborhoodDescent
from .Greedy import GreedySPlex
import numpy.typing as npt
class GRASP:
    def __init__(self, neighborhood: Neighborhood, trials = 100) -> None:
        self.neighborhood = neighborhood
        self.local_search = VariableNeighborhoodDescent([neighborhood])
        self.trials = 100
        self.solutions_found = list()

    def search(self, s:int, A: npt.NDArray[np.int_], W: npt.NDArray[np.int_]) -> Solution:
        greedy_search = GreedySPlex(A, W, s=s)
        for _ in range(self.trials):
            A1, splexes = greedy_search.random_solution()

            x0 = Solution(A,W,A1,splexes)
            x1 = self.local_search.search(x0)
            self.solutions_found.append(x1)

        return min(self.solutions_found, key=lambda x: x.obj())