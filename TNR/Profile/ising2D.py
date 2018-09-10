import numpy as np

from TNR.Models.isingModel import IsingModel2Dopen
from TNR.Contractors.mergeAllContractor import mergeContractor
from TNR.Contractors.heuristics import entropyHeuristic

nX = 6
nY = 6

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
    print('Ratio:',np.log(nn.tensor.array) / (nX * nY * 0.93))
