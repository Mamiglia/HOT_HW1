import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood
from typing import Collection

class VariableNeighborhoodDescent:
    def __init__(self, neighborhoods : Collection[Neighborhood], max_it: int = 100) -> None:
        self.neighborhoods = neighborhoods
        self.max_it = max_it

    def search(self, x0 : Solution) -> Solution:
        x = x0
        it = 0
        for neighborhood in self.neighborhoods:
            it += 1
            xs = neighborhood.neighbor_list(x)
            x1 = min(xs, key = lambda x: x.obj())

            if x1.obj() < x.obj():
                x = x1
                break
        else:
            return self.search(x)
        
        return x
                