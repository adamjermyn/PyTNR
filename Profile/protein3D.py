import numpy as np

from TNRG.Models.protein_aggregation_3d import PA3D
from TNRG.Contractors.mergeContractor import mergeContractor

nX = 3
nY = 3
nZ = 3

accuracy = 1e-5

h = 0.1
J = 0.5
q = 0.2

n = PA3D(nX, nY, nZ, h, J, q, accuracy)
n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=2)

print(len(n.nodes))
for nn in n.nodes:
	print(nn.tensor.array)
