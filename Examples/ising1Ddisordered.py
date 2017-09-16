import numpy as np
import time

from TNR.Models.isingModel import IsingModel1Ddisordered
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import entropyHeuristic


def ising1DFreeEnergy(nX, h, J, accuracy):
    n = IsingModel1Ddisordered(nX, h, J, accuracy)
    n = mergeContractor(
        n,
        accuracy,
        entropyHeuristic,
        optimize=False,
        merge=False,
        plot=False)
    return n.array[1] / nX


h = 1
J = 1

size = [
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    150,
    200,
    250,
    300]
size = 3 * size
accuracy = 1e-3
res = []

for s in size:
    print(s)
    start = time.clock()
    f = ising1DFreeEnergy(s, h, J, accuracy)
    end = time.clock()
    res.append((s, f, end - start))

res = np.array(res)
np.savetxt('ising1Dh_disordered.dat', res)
