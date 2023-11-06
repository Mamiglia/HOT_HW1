import numpy as np
from typing import Dict, Tuple

def obj_function(A0 : np.ndarray, A1: np.ndarray, W : np.ndarray):
    X = (A0 != A1)
    return (X * W).sum()

# def obj_function(X: np.ndarray, W : np.ndarray):
#     return (X * W).sum()

def adjmatrix2adjmap(W: np.ndarray) -> Dict[Tuple[int, int], int]:
    return {(i,j): w for i,a in enumerate(W) for j,w in enumerate(a) if i<j} 