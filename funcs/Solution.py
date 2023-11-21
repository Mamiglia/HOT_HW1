import numpy as np
import numpy.typing as npt


import dataclasses as dc
from copy import deepcopy
from typing import Collection, Dict


@dc.dataclass
class Solution:
    A: npt.NDArray[np.int_]
    W: npt.NDArray[np.int_]
    A1: npt.NDArray[np.int_]
    clusters: Dict[int, Collection[int]]
    _obj: int = -1

    @property
    def size(self):
        return self.A.shape[0]

    def copy(self):
        return Solution(self.A, self.W, self.A1.copy(), deepcopy(self.clusters))

    def obj(self) -> int:
        if self._obj != -1:
            return self._obj

        self._obj = ((self.A1 != self.A) * self.W).sum()
        return self._obj