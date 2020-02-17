import numpy as np
import time

from TNR.Models.isingModel import IsingModel2DdisorderedProbs
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic
from TNR.Actions.swap_elim import swap_elim as eliminateLoops

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def ising2DFreeEnergy(nX, nY, h, J, accuracy):
    n, buckets = IsingModel2DdisorderedProbs(nX, nY, h, J, accuracy)
    n = mergeContractor(
        n,
        accuracy,
        heuristic,
        optimize=True,
        merge=False)
    return n, buckets

size = (8,8)

h = 0.2
J = 0.5
accuracy = 1e-5

start = time.clock()
n, buckets = ising2DFreeEnergy(size[0], size[1], h, J, accuracy)
end = time.clock()

import pickle
pickle.dump([n, buckets], open('2DIsingDisorderedProbs.net', 'wb'))

print(n)

print(size[0] * size[1], end - start)

n = n.nodes.pop()
t = n.tensor

indices = []

for i in range(len(buckets)):
    indices.append([])
    for j in range(len(buckets[i])):
        indices[i].append(n.buckets.index(buckets[i][j]))

def traceOut2(tens, ind0, ind1):
    while tens.rank > 2:
        i = np.random.randint(tens.rank)
        if i != ind0 and i != ind1:
            tens = tens.traceOut(i)
            if i < ind1:
                ind1 -= 1
            if i < ind0:
                ind0 -= 1
    return tens

twop = []

for i in range(len(buckets)):
    twop.append([])
    for j in range(len(buckets[i])):
        if i == j:
            twop[i].append(np.identity(2))
        else:
            twop[i].append(traceOut2(t, indices[0][0], indices[i][j]).array)
            twop[i][j] /= np.sum(twop[i])
            print('-----')
            print(i,j)
            print(twop[i][-1])
