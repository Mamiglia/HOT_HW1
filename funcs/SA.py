import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood

class SimulatedAnnealing:
    def __init__(self, init_temperature: float, neighborhood : Neighborhood, cooling : float = 0.95 ) -> None:
        self.T0 = init_temperature
        self.cooling = cooling
        self.neighborhood = neighborhood

    def search(self, x0:Solution) -> Solution:
        T = self.T0
        x = x0
        best = x

        epoch = 0
        while not SimulatedAnnealing.stop_criterion(epoch):
            iteration = 0
            while not SimulatedAnnealing.equilibrium(iteration):
                x1 = self.neighborhood.shaking(x)

                p = np.exp((x.obj() - x1.obj())/T)

                if p>=1 or p > np.random.random():
                    x = x1
                    if x.obj() < best.obj():
                        best = x

                iteration+=1
            
            epoch+=1
            T = T/self.cooling
        return best

                

    @staticmethod
    def equilibrium(iteration : int) -> bool:
        return iteration > 5000

    @staticmethod
    def stop_criterion(epoches : int, max_epoches : int = 20) -> bool:
        return epoches > max_epoches