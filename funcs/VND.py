import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood
from typing import Collection

class VariableNeighborhoodDescent:
    def __init__(self, neighborhoods : Collection[Neighborhood]) -> None:
        self.neighborhoods = neighborhoods

        self.history = []

    def search(self, x0 : Solution) -> Solution:
        x = x0
        improved = True
        while improved:
            improved = False
            for neighborhood in self.neighborhoods:
                # xs = neighborhood.neighbor_list(x)
                # if len(xs) == 0:
                #     continue
                # x1 = min(xs, key = lambda x: x.obj())

                # if x1.obj() < x.obj():
                #     print(f'Found {len(self.history)}th improvement at {neighborhood}. Delta={x.obj()-x1.obj()}')
                #     x = x1
                #     improved = True
                #     self.history.append(x1.obj())
                #     break

                # First Improvement
                for x1 in neighborhood.neighbors(x):
                    if x1.obj() < x.obj():
                        # print(f'Found {len(self.history)}th improvement at {neighborhood}. Delta={x.obj()-x1.obj()}')
                        x = x1
                        improved = True
                        self.history.append(x1.obj())
                        break
                else:
                    break            
                

        return x
                