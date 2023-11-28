from typing import Iterator, List, Tuple, Dict
from abc import abstractmethod, ABCMeta
import numpy as np
from funcs.Solution import Solution
from funcs import is_splex
import random
import numpy.typing as npt
from itertools import combinations, chain

class Neighborhood(metaclass=ABCMeta):
    @abstractmethod
    def shaking(self, x : Solution) -> Solution:
        '''pick a random neighbor'''
        pass

    @abstractmethod
    def neighbors(self, x : Solution) -> Iterator[Solution]:
        '''return an iterator over all the possible neighbors'''
        pass

    def neighbor_list(self, x: Solution) -> List[Solution]:
        return list(self.neighbors(x))


class NeighborhoodUnion(Neighborhood):
    def __init__(self, *neighborhoods: Neighborhood) -> None:
        self.neighborhoods = neighborhoods

    def shaking(self, x: Solution) -> Solution:
        return random.choice(self.neighborhoods).shaking(x)
    
    def neighbors(self, x: Solution) -> Iterator[Solution]:
        return chain.from_iterable(neigh.neighbors(x) for neigh in self.neighborhoods)

class Flip1(Neighborhood):
    '''Flips a single edge inside a plex'''
    def __init__(self, s:int) -> None:
        self.S = s

    def neighbors(self, x: Solution) -> Iterator[Solution]:
        degree = x.A1.sum(axis=-1)
        for i in range(x.size):
            plex = x.clusters[i]
            min_edges = len(plex) - self.S

            for j in plex:
                edge_removal = x.A1[i,j] == 1
                min_degree = min(degree[i], degree[j])

                if i >= j or (edge_removal and min_degree <= min_edges):
                    continue
                yield self.step(x,i,j)
    
    def shaking(self, x : Solution) -> Solution:
        plex = random.choice([plex for plex in x.clusters.values() if len(plex)>=2])
        i,j  = random.sample(plex, 2)
        min_edges = len(plex) - self.S
        if x.A1[i,j] == 1 and min(x.A1[i].sum(), x.A1[j].sum()) <= min_edges:
            return x
                
        return self.step(x, i, j)

    def step(self, x : Solution, i: int, j: int) -> Solution:
        y = x.copy()
        y.A1[i,j] = 1 - y.A1[i,j]
        y.A1[j,i] = y.A1[i,j] 
        
        # Delta evaluation
        y._obj = x.obj() + (y.A1[i,j]*2-1)*x.W[i,j]
        return y
    
  
class QualityFlip1(Flip1):
    '''Flips a single edge inside a plex if there's an improvement'''
    def neighbors(self, x: Solution) -> Iterator[Solution]:
        degree = x.A1.sum(axis=-1)
        for i in range(x.size):
            plex = x.clusters[i]
            min_edges = len(plex) - self.S

            for j in plex:
                positive_gain = ((2*x.A1[i,j]-1) * x.W[i,j]) > 0
                if i >= j or positive_gain or (x.A1[i,j] == 1 and min(degree[i], degree[j]) <= min_edges):
                    continue
                yield self.step(x,i,j)
    
    def shaking(self, x : Solution) -> Solution:
        plex = random.choice([plex for plex in x.clusters.values() if len(plex)>=2])
        plex_idx = np.ix_(plex, plex)
        plex_m = x.A1[plex_idx]
        degree = plex_m.sum(axis=-1)
        min_edges = len(plex) - self.S

        # print(degree>min_edges + plex_m==0)
        enough_degree = ((degree>min_edges)[:,np.newaxis] + plex_m==0)
        convenient    = ((2*plex_m-1) * x.W[plex_idx] > 0) 
        feasible_edges = enough_degree * convenient

        ii,jj = np.where(feasible_edges >= 1)
        if ii.shape[0] == 0:
            return x
        k = np.random.randint(ii.shape[0])
        i,j = plex[ii[k]], plex[jj[k]]
        return self.step(x, i, j)



class SwapEdge(Neighborhood):
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
        vicini = np.where(x.A1[s])[0]
        k = random.choice(vicini)

        return self.swap(x, i, j, s, k)
    
    def swap(self, x, i, j, s, k):
        y = x.copy()
        y.A1[i,j] = 0
        y.A1[j,i] = 0
        y.A1[i,s] = 1
        y.A1[s,i] = 1
        y.A1[j,k] = 1
        y.A1[k,j] = 1

        return y
    
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

class SwapNode(Neighborhood):
    '''Swaps the edges of 2 nodes'''
    def __init__(self, size : int) -> None:
        self.all_nodes = list(range(size))

    def shaking(self, x:Solution) -> Solution:
        i,j = random.sample(self.all_nodes, 2)

        return self.step(x,i,j)
    
    def neighbors(self, x:Solution) -> Iterator[Solution]:
        for i in range(x.size):
            for j in range(i):
                yield self.step(x,i,j)
    
    def step(self, x:Solution, i: int, j:int) -> Solution:
        y = x.copy()
        y.A1[[i,j]] = y.A1[[j,i]]
        y.A1[:, [i,j]] = y.A1[:, [j,i]]

        if y.clusters[i] is not y.clusters[j]:
            y.clusters[i], y.clusters[j] = y.clusters[j], y.clusters[i]
            y.clusters[i].remove(j)
            y.clusters[i].append(i)
            y.clusters[j].remove(i)
            y.clusters[j].append(j)

        # Delta Eval
        y._obj = x.obj() + x.W[i]@(y.A1[i] - x.A1[i]) + x.W[j]@(y.A1[j] - x.A1[j]) 

        return y

from funcs import show_adj_matrix

class MoveNode(Neighborhood):
    '''Moves a node from a cluster to another'''
    def __init__(self, s: int, size : int) -> None:
        self.all_nodes = list(range(size))
        self.S = s

    def neighbors(self, x: Solution) -> Iterator[Solution]:
        degree = x.A1.sum(axis=-1)
        for i in self.all_nodes:
            for plex in self.get_cluster_list(x.clusters):
                if i in plex:
                    yield self.step(x,i,None)
                else:
                    yield self.step(x,i,plex, degree[plex])


    def shaking(self, x: Solution) -> Solution:
        i = random.choice(self.all_nodes)
        
        plexes = self.get_cluster_list(x.clusters)
        plex = random.choice(plexes)
        if i in plex:
            return self.step(x,i,None)
        
        return self.step(x, i, plex, x.A1[plex].sum(axis=-1))


    def get_cluster_list(self, clusters_dict : Dict[int, List[int]]) -> List[List[int]]:
        clusters = []
        clusters_ids = set()
        for plex in clusters_dict.values():
            if id(plex) in clusters_ids:
                continue
            clusters.append(plex)
            clusters_ids.add(id(plex))

        return clusters

    def step(self, x : Solution, i: int, plex: List[int]= None, degree_plex:npt.NDArray[np.int32]=None) -> Solution:
        y = x.copy()
        # Remove i from current cluster
        y.A1[i] = 0
        y.A1[:,i] = 0
        y.clusters[i].remove(i)
        y._obj = x.obj() - (x.A1[i]*y.W[i]).sum()

        if plex is None:
            # Create new cluster
            y.clusters[i] = [i]
            assert is_splex(y.A1, self.S)

            return y
        
        # Attach i to the new cluster by selecting 'min_edges' edges
        min_edges = max(1, len(plex) + 1 - self.S)
        cost = x.W[i,plex].copy()
        # Select first all nodes *needing* to have a new edge, then those with minimum cost
        needy_nodes = np.where(degree_plex < min_edges)[0]
        cost[needy_nodes] = 9999999
        min_cost_edges = []
        if len(needy_nodes) < min_edges:
            missing_edges = min_edges-len(needy_nodes)
            min_cost_edges = np.argpartition(cost, missing_edges-1)[:missing_edges]

        jj = [plex[v] for v in chain(needy_nodes, min_cost_edges)]

        y.A1[i,jj] = 1
        y.A1[jj,i] = 1

        y.clusters[i] = y.clusters[plex[0]]
        y.clusters[i].append(i)
        y._obj += (y.W[i,jj]).sum()

        assert is_splex(y.A1, self.S)

        return y


import numba as nb
from .Greedy import weighted_karger
# @nb.jitclass
class Divide(Neighborhood):
    def shaking(self, x: Solution) -> Solution:
        plex = random.choice(x.clusters)
        return self.step(x, plex)

    def step(self, x: Solution, plex:List[int]):
        y = x.copy()
        plex_idx = np.ix_(plex,plex)

        subplexes = weighted_karger(x.W[plex_idx])
        subplexes = [[plex[i] for i in subplex] for subplex in subplexes]

        mask = np.ones_like(y.A1, dtype=np.int32)
        for subplex in subplexes:
            subplex_idx = np.ix_(subplex, subplex)
            mask[subplex_idx[0]] = 0
            mask[:, subplex_idx[1]] = 0
            mask[subplex_idx] = 1
            for i in subplex:
                y.clusters[i] = subplex
        y.A1 *= mask

        y._obj = x.obj() - ((1-mask) * x.A1 * x.W).sum()//2
        return y 
    
    def neighbors(self, x: Solution) -> Iterator[Solution]:
        for plex in x.clusters:
            yield self.step(x,plex)
                

