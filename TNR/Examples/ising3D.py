import numpy as np
import time
import sys

from TNR.Models.isingModel import IsingModel3Dopen
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def ising3DFreeEnergy(nX, nY, nZ, h, J, accuracy):
    n = IsingModel3Dopen(nX, nY, nZ, h, J, accuracy)
    n = mergeContractor(
        n,
        accuracy,
        heuristic,
        optimize=True,
        merge=False,
        plot=False)
    return n.array[1] / (nX * nY * nZ)


J = float(sys.argv[1])

h = 0
accuracy = 1e-3
size = [(5,5,5)]

res = []

for s in size:
    logger.info(
        'Examining system of size ' +
        str(s) +
        ' and J = ' +
        str(J) +
        '.')
    start = time.clock()
    f = ising3DFreeEnergy(s[0], s[1], s[2], h, J, accuracy)
    end = time.clock()
    res.append((s[0] * s[1] * s[2], f, end - start))

res = np.array(res)

print(res)