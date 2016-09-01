import sys
sys.path.append('../TensorNetwork/')
sys.path.append('../Models/')
import numpy as np
from isingModel2D import IsingModel, exactIsing
import cProfile

nX = 10
nY = 10

for j in np.linspace(-1,1,num=10):
	network, lattice = IsingModel(nX, nY, 0, j)
	network.contract(mergeL=True, compressL=True, eps=1e-4)
	arr, logS, _ = network.topLevelRepresentation()
	q = np.log(arr) + logS
	q /= (nX*nY)
	print(j,q,exactIsing(j))
