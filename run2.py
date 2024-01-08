import os
from funcs import *
import frigidum

from funcs import readin, Solution, GA, is_splex
from funcs.Greedy import Karger
import numpy as np
from funcs.readin import writeout

# folder = 'data/test_instances/'
folder = 'data\inst_competition'

for file in os.listdir(folder):
    if not os.path.isfile(folder + file):
        continue
    prob_name = file.split('.')[0]
    print(prob_name)
    S,A,W = readin(file)
    greedy = Karger(A, W, S)
    def random_start():
        A1, splexes = greedy.random_solution()
        return Solution.build(A,W,A1, splexes)


    assert(is_splex(xb.A1, S))
    print(xb.obj())

    details = f'_grasp_{xb.obj()}'

    writeout(A, xb.A1, folder, prob_name, details)