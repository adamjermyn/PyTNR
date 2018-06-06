import numpy as np

from TNR.Models.protein_aggregation_3d import PA3D
from TNR.Contractors.mergeContractor import mergeContractor

nX = 4
nY = 5
nZ = 5

accuracy = 1e-3

h = 2
J = 2
q = 1

n = PA3D(nX, nY, nZ, h, J, q, accuracy)
n = mergeContractor(
    n,
    accuracy,
    optimize=False,
    merge=False,
    verbose=2,
    mergeCut=20)

print(len(n.nodes))
for nn in n.nodes:
    print(np.log(nn.tensor.array))
