import sys
sys.path.append('../TensorNetwork/')
sys.path.append('../Models/')
import numpy as np
from isingModel import IsingModel2D, exactIsing2D, IsingModel3D
import cProfile

nX = 10
nY = 10

for j in np.linspace(-1,1,num=10):
	network, lattice = IsingModel2D(nX, nY, 0, j)
	network.contract(mergeL=True, compressL=True, eps=1e-4)
	arr, logS, _ = network.topLevelRepresentation()
	q = np.log(arr) + logS
	q /= (nX*nY)
	print(j,q,exactIsing2D(j))

nX = 10
nY = 10
nZ = 10

network, lattice=  IsingModel3D(nX, nY, nZ, 0.5, 0.5)
network.contract(mergeL=True, compressL=True, eps=1e-4)
arr, logS, _ = network.topLevelRepresentation()
q = np.log(arr) + logS
print(q)
