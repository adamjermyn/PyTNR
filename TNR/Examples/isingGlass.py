import numpy as np
import time
import sys

from TNR.Models.isingModel import IsingSpinGlass
from TNR.Contractors.contractor import contractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def isingFreeEnergy(nX, J, k, accuracy):
    n = IsingSpinGlass(nX, J, k, accuracy)

    c = contractor(n)
    done = False
    while not done:
        node, done = c.take_step(heuristic, eliminateLoops=True)
        c.optimize(node)
    n = c.network

    arr, log_arr, bdict = n.array
    return (np.log(np.abs(arr)) + log_arr) / nX

J = 1
accuracy = 1e-6
k = 1.5

res = []

s = int(sys.argv[1])

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

np.savetxt('isingGlass_' + str(s) + '_' + sys.argv[2] + '.dat', res)
