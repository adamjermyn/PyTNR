import numpy as np
import sys
from TNR.Models.isingModel import IsingModel3Dopen as IsingModel3D
from TNR.Contractors.mergeAllContractor import mergeContractor

n = int(sys.argv[1])

nX = n
nY = n
nZ = n

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
