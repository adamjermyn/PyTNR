import numpy as np

from TNR.Models.isingModel import IsingModel3Dopen as IsingModel3D
from TNR.Contractors.mergeAllContractor import mergeContractor

nX = 6
nY = 6
nZ = 5

accuracy = 1e-4

h = 0.1
J = 0.5

n = IsingModel3D(nX, nY, nZ, h, J, accuracy)
n = mergeContractor(
    n,
    accuracy,
    optimize=False)

print(len(n.nodes))
for nn in n.nodes:
    print(nn.tensor.array)
