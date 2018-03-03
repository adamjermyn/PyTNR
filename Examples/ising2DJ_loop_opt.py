import numpy as np
import time
import sys

from TNR.Models.isingModel import IsingModel2D, exactIsing2D
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def ising2DFreeEnergy(nX, nY, h, J, accuracy):
    n = IsingModel2D(nX, nY, h, J, accuracy)
    n = mergeContractor(
        n,
        accuracy,
        heuristic,
        optimize=True,
        merge=False,
        plot=False)
    return n.array[1] / (nX * nY)


J = float(sys.argv[1])

h = 0
accuracy = 1e-3
size = [(2,2)]

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
