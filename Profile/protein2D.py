import numpy as np

from TNR.Models.protein_aggregation_2D import PA2D
from TNR.Contractors.mergeContractor import mergeContractor

nX = 40
nY = 40

accuracy = 1e-5

h = 0.1
J = 0.5
q = 0.1

n = PA2D(nX, nY, h, J, q, accuracy)
n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=2)

print(len(n.nodes))
for nn in n.nodes:
	print(nn.tensor.array)
