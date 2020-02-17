import numpy as np
import time

from TNR.Models.isingModel import IsingModel1D, exactIsing1DJ
from TNR.Contractors.contractor import contractor
from TNR.Contractors.heuristics import entropyHeuristic as heuristic


def ising1DFreeEnergy(nX, h, J, accuracy):
    n = IsingModel1D(nX, h, J, accuracy)

    c = contractor(n)
    done = False
    while not done:
        node, done = c.take_step(heuristic, eliminateLoops=True)
    n = c.network

    arr, log_arr, bdict = n.array
    return (np.log(np.abs(arr)) + log_arr) / nX


for J in [-2, -1, -0.5, 0, 0.5, 1, 2]:
    size = list(range(2, 25))

    h = 0
    accuracy = 1e-6

    res = []

    for s in size:
        start = time.clock()
        f = ising1DFreeEnergy(s, h, J, accuracy)
        end = time.clock()
        res.append((s, f, f - exactIsing1DJ(s, J), end - start))

    res = np.array(res)

    np.savetxt('ising1DJ_J=' + str(J) + '.dat', res)
