import numpy as np

from TNR.Models.isingModel import IsingModel3D
from TNR.Contractors.mergeContractor import mergeContractor

nX = 6
nY = 6
nZ = 6

accuracy = 1e-3

h = 0.1
J = 3.0

n = IsingModel3D(nX, nY, nZ, h, J, accuracy)
n = mergeContractor(
    n,
    accuracy,
    optimize=True,
    merge=True,
    verbose=2,
    mergeCut=10)

print(len(n.nodes))
for nn in n.nodes:
    print(nn.tensor.array)
