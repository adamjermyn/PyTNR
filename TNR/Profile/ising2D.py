import numpy as np

from TNR.Models.isingModel import IsingModel2Dopen
from TNR.Contractors.mergeAllContractor import mergeContractor
from TNR.Contractors.heuristics import entropyHeuristic
import sys

n = sys.argv[1]
n = int(n)
nX = n
nY = n

accuracy = 1e-4

h = 0.0
J = 0.43

n = IsingModel2Dopen(nX, nY, h, J, accuracy)
n = mergeContractor(
    n,
    accuracy,
    entropyHeuristic,
    optimize=False)

print(len(n.nodes))
for nn in n.nodes:
    print('Ratio:', nn.tensor.logNorm / (nX * nY * 0.93))
