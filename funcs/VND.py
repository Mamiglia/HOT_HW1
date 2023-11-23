import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood
from typing import Collection

class VariableNeighborhoodDescent:
    def __init__(self, neighborhoods : Collection[Neighborhood]) -> None:
        self.neighborhoods = neighborhoods

    def search(self, x0 : Solution) -> Solution:
        x = x0
        improved = True
        while improved:
            improved = False
            for neighborhood in self.neighborhoods:
                xs = neighborhood.neighbor_list(x)
                x1 = min(xs, key = lambda x: x.obj())

                if x1.obj() < x.obj():
                    x = x1
                    improved = True
                    break


        
        return x
                