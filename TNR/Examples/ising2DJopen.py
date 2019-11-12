import numpy as np
import time
import sys

from TNR.Models.isingModel import IsingModel2Dopen, exactIsing2D
from TNR.Contractors.managedContractor import managedContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def ising2DFreeEnergy(nX, nY, h, J, accuracy):
    n = IsingModel2Dopen(nX, nY, h, J, accuracy)
    n = managedContractor(
        n,
        3,
        accuracy,
        heuristic,
        optimize=True,
        cost_cap = 1e9)

    arr, log_arr, bdict = n.array
    return (np.log(arr) + log_arr) / (nX * nY)


accuracy = 1e-6
h = 0
J = float(sys.argv[1])
nX = int(sys.argv[2])
nY = int(sys.argv[3])
s = (nX, nY)

start = time.clock()
f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
end = time.clock()
res = (s[0] * s[1], f, f - exactIsing2D(J), end - start)

np.savetxt('ising2DJ_open_' + str(J) + '_' + str(nX) + '_' + str(nY) + '.dat', res)
