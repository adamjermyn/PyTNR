import numpy as np
import time
import sys

from TNR.Models.isingModel import IsingSpinGlass
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def isingFreeEnergy(nX, J, k, accuracy):
    n = IsingSpinGlass(nX, J, k, accuracy)
    n = mergeContractor(
        n,
        accuracy,
        heuristic,
        optimize=True,
        merge=False,
        plot=False)

    arr, log_arr, bdict = n.array
    return (np.log(np.abs(arr)) + log_arr) / nX

J = 1
accuracy = 1e-6
k = 1.5

res = []

size = int(sys.argv[1])

logger.info(
    'Examining system of size ' +
    str(s) +
    ' and J = ' +
    str(J) +
    '.')
start = time.clock()
f = isingFreeEnergy(s, J, k, accuracy)
end = time.clock()
res.append((s, f, end - start))

res = np.array(res)

np.savetxt('isingGlass_' + str(size) + '_' + sys.argv[2] + '.dat', res)
