import numpy as np

from TNR.Models.isingModel import IsingModel2Dopen
from TNR.Contractors.mergeContractor import mergeContractor

nX = 3
nY = 3

accuracy = 1e-3

h = 0.1
J = 0.5

n = IsingModel2Dopen(nX, nY, h, J, accuracy)
n = mergeContractor(
    n,
    accuracy,
    optimize=False,
    merge=False,
    mergeCut=15)

print(len(n.nodes))
for nn in n.nodes:
    print(nn.tensor.array)
