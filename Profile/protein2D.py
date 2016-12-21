import numpy as np

from TNRG.Models.protein_aggregation_2D import PA2D
from TNRG.Contractors.mergeContractor import mergeContractor

nX = 10
nY = 10

accuracy = 1e-5

h = 0.1
J = 0.5

n = PA2D(nX, nY, h, J, accuracy)
n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=2)

print(len(n.nodes))
for nn in n.nodes:
	print(nn.tensor.array)
