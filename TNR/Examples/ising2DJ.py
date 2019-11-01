import numpy as np
import time
import sys

from TNR.Models.isingModel import IsingModel2D, exactIsing2D
from TNR.Contractors.managedContractor import managedContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def ising2DFreeEnergy(nX, nY, h, J, accuracy):
    n = IsingModel2D(nX, nY, h, J, accuracy)
    n = managedContractor(
        n,
        3,
        accuracy,
        heuristic,
        optimize=False,
        cost_cap = 1e6)
    return n.array[1] / (nX * nY)



for J in [-2, -1, -0.5, 0, 0.5, 1, 2]:

    h = 0
    accuracy = 1e-3
    size = [(2, 2), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4), (4, 4), (3, 6), (4, 5), (3, 7), (3, 8),
            (5, 5), (3, 9), (4, 7), (5, 6), (4, 8), (5, 7), (6, 6), (6, 7), (7, 7)]

    res = []

    for s in size:
        logger.info(
            'Examining system of size ' +
            str(s) +
            ' and J = ' +
            str(J) +
            '.')
        start = time.clock()
        f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
        end = time.clock()
        res.append((s[0] * s[1], f, f - exactIsing2D(J), end - start))

    res = np.array(res)

    print(res)

np.savetxt('ising2DJ_J=' + str(J) + '.dat', res)
