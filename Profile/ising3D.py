import numpy as np

from TNRG.Models.isingModel import IsingModel3D
from TNRG.Contractors.mergeContractor import mergeContractor

nX = 4
nY = 4
nZ = 4

accuracy = 1e-5

h = 0.1
J = 0.5

n = IsingModel3D(nX, nY, nZ, h, J, accuracy)
n = mergeContractor(n, accuracy, optimize=False, merge=False, verbose=2)

print(len(n.nodes))
for nn in n.nodes:
	print(nn.tensor.array)
