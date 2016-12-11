import sys 
sys.path.append('../Models/') 
sys.path.append('../Contractors/') 
import numpy as np
from compress import compressLink
from isingModel import IsingModel1D, IsingModel2D, exactIsing2D
from mergeContractor import mergeContractor

print('Testing h.')

nX = 4
accuracy = 1e-10
h = 1.0
J = 0.0

n = IsingModel1D(nX, h, J, accuracy)

n = mergeContractor(n, accuracy, optimize=False, merge=False, verbose=0)

assert len(n.nodes) == 1

nn = n.nodes.pop()

print(np.log(nn.tensor.array)/nX)
print(np.log(np.cosh(h)*2))

print('Testing J.')

nX = 5
nY = 5
accuracy = 1e-5
h = 0.0
J = 3.0

n = IsingModel2D(nX, nY, h, J, accuracy)

n = mergeContractor(n, accuracy, optimize=True, merge=False, verbose=2)

assert len(n.nodes) == 1

nn = n.nodes.pop()

print(np.log(nn.tensor.array)/(nX*nY))
print(exactIsing2D(J))

