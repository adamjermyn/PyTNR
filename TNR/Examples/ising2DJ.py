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
        optimize=True,
        cost_cap = 1e9)
    return n.array[1] / (nX * nY)

accuracy = 1e-3
h = 0
J = float(sys.argv[1])
nX = int(sys.argv[2])
nY = int(sys.argv[3])
s = (nX, nY)

start = time.clock()
f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
end = time.clock()
res = (s[0] * s[1], f, f - exactIsing2D(J), end - start)

np.savetxt('ising2DJ_periodic_' + str(J) + '_' + str(nX) + '_' + str(nY) + '.dat', res)
