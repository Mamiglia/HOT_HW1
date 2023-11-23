from typing import Iterator, List
from abc import abstractmethod, ABCMeta
import numpy as np
from funcs.Solution import Solution
import random

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
        for i, cluster in x.clusters.items():
            for j in cluster:
                if i == j:
                    continue
                yield self.step(x,i,j)
    
    def shaking(self, x : Solution) -> Solution:
        i = np.random.randint(x.size)
        j = i
        while i==j and len(x.clusters[i]) > 1:
            j = random.choice(x.clusters[i])
        return self.step(x, i, j)

    def step(self, x : Solution, i: int, j: int) -> Solution:
        x_n = x.copy()
        x_n.A1[i,j] = 1 - x_n.A1[i,j]
        x_n.A1[j,i] = x_n.A1[i,j] 

        # Delta evaluation
        x_n._obj = x.obj() + ((x_n.A[i,j] != x_n.A1[i,j])*2-1)*x.W[i,j]
        return x_n

class Swap(Neighborhood):
    def shaking(self, x: Solution) -> Solution:
        possible_clusters = [k for k,v in x.clusters.items() if len(v) >= 4]
        if len(possible_clusters) == 0:
            return x
        i = random.choice(possible_clusters)
        vicini = np.where(x.A1[i])[0]
        if vicini.shape[0] == 0:
            return self.shaking(x)
        j = random.choice(vicini)
        s = random.choice(x.clusters[i])
        while s == j or s==i:
            s = random.choice(x.clusters[i])
        vicini = np.where((x.A1[s])*(1-x.A1[i])*(1-x.A1[j]))[0]
        k = random.choice(vicini)

        return self.swap(x, i, j, s, k)
    
    def swap(self, x, i, j, s, k):
        x_n = x.copy()
        x_n.A1[i,j] = 0
        x_n.A1[j,i] = 0
        x_n.A1[i,s] = 1
        x_n.A1[s,i] = 1
        x_n.A1[j,k] = 1
        x_n.A1[k,j] = 1

        return x_n
    
    def neighbors(self, x: Solution) -> Iterator[Solution]:
        for i in range(x.size):
            vicini_i = np.where(x.A1[i])[0]
            for j in vicini_i:
                for s in x.clusters[i]:
                    if s == i or s==j:
                        continue
                    vicini_j = np.where((x.A1[s])*(1-x.A1[i])*(1-x.A1[j]))[0]
                    for k in vicini_j:
                        yield self.swap(x, i, j, s, k)
