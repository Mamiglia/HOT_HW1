import numpy as np
from funcs import adjmatrix2adjmap
from itertools import chain
import random

class GreedySPlex:
    TERROR_VALUE = 99999
    def __init__(self, adjacency_matrix : np.ndarray, weight_matrix: np.ndarray, s: int):
        self.A = adjacency_matrix
        self.W = weight_matrix.copy()
        np.fill_diagonal(self.W, GreedySPlex.TERROR_VALUE)
        self.W[adjacency_matrix==1] *= -1
        self.s = s 
        self.weightmap = adjmatrix2adjmap(self.W)

    def random_solution(self):
        # We start with the graph with all edges removed
        # and we try to add them back one at a time
        edges = [e for e in self.weightmap.keys() if self.weightmap[e]<0]
        random.shuffle(edges)
        return self.solve(edges)

    def solution(self):
        # We start with the graph with all edges removed
        # and we try to add them back one at a time (sorted by weight)
        edges = [e for e in self.weightmap.keys() if self.weightmap[e]<0]
        edges_by_weight = sorted(edges, key=self.weightmap.get)
        return self.solve(edges_by_weight)

    def solve(self, edges: dict):
        splexes = {i:{i} for i in range(self.A.shape[0])}
        A1 = np.zeros(self.A.shape, dtype=int)

        for i,j in edges:
            # print(f"Considering [{i},{j}]")
            s_i = splexes[i]
            s_j = splexes[j]
            plex_size = len(s_i) + len(s_j)
            
            if s_i is s_j or plex_size <= (self.s+1):
                # print('\tFast merge.')
                # if they are less than S+1 they are always a legit s-plex
                self.merge(i,j,splexes, A1)
                continue

            cost = 0
            added_edges = dict()
            candidate_splex = list(chain(s_i, s_j))

            # We now try to see if merging the 2 splexes is convenient or not
            # For each of the nodes we check how many edges we need to add to make it a valid splex
            # we then add the "less expensive" edges
            # if the operation is convinient we perform it, otherwise we abort it

            W1 = self.W.copy()
            # in order to constrain our choice of edges to only those in the considered splexes,
            # We're going to use a Terror_Value (very high number) to make the algorithm pick only the desired edges

            # set the weights of candidate plex to very low value in order to make the algorithm choose that
            W1[np.ix_(candidate_splex, candidate_splex)] -= GreedySPlex.TERROR_VALUE
            # W1[list(s_j), list(s_i)] -= GreedySPlex.TERROR_VALUE
            # set the weights of already present edges to terror value
            W1[A1 == 1] = GreedySPlex.TERROR_VALUE
            np.fill_diagonal(W1, GreedySPlex.TERROR_VALUE)
            # print(f'\tCandidate Merging {s_i} U {s_j}')
            missing_edges = {k:plex_size - self.s - A1[k].sum() for k in candidate_splex}
            for k in candidate_splex:
                m = missing_edges[k]
                # print(f'\t\tConsidering {k}, missing: {m}')
                if m <= 0:
                    added_edges[k] = []
                    continue

                weights = W1[k]
                nodes = np.argpartition(weights, m)[:m]
                # print(f'\t\t{weights}')
                # print(f'\t\t{nodes}')

                added_edges[k] = nodes
                for node in nodes:
                    missing_edges[node] -= 1
                cost += self.W[k,nodes].sum()
                W1[k,nodes] = GreedySPlex.TERROR_VALUE
                W1[nodes,k] = GreedySPlex.TERROR_VALUE

            # if the computed cost indicates that there's a gain
            # print(f'\tCost: {cost}')
            if cost < 0:
                # print('\tMerge.')
                self.merge(i,j,splexes, A1, added_edges)

        return A1, splexes
    
    def merge(self, i: int, j: int, splexes: dict, A1: np.ndarray, added_edges : dict = None):
        # Set edge
        A1[i,j] = 1
        A1.T[i,j] = 1
        # Merge splexes
        if splexes[i] is not splexes[j]:
            splexes[i] = splexes[i] | splexes[j] # union
            for k in splexes[i]:
                splexes[k] = splexes[i]

                if added_edges is not None:
                    A1[k, added_edges[k]] = 1
                    A1[added_edges[k], k] = 1