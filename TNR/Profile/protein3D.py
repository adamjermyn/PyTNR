import numpy as np

from TNR.Models.protein_aggregation_3d import PA3D
from TNR.Contractors.mergeAllContractor import mergeContractor

nX = 4
nY = 4
nZ = 4

accuracy = 1e-4

h=2.
J=2.
q=1.

n = PA3D(nX, nY, nZ, h, J, q, accuracy)
n = mergeContractor(
    n,
    accuracy,
    optimize=False)

print(len(n.nodes))
for nn in n.nodes:
    print(nn.tensor.logNorm)
