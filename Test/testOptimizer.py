import sys
sys.path.append('../TensorNetwork/')
sys.path.append('../Models/')
import numpy as np
from isingModel import IsingModel2D, exactIsing2D, IsingModel3D, RandomIsingModel2D
import cProfile

nX = 15
nY = 15

network, lattice = IsingModel2D(nX, nY, 0.1, 0.5)
network.contract(mergeL=True, compressL=True, eps=1e-4)
network.optimize(mergeL=True, compressL=True, eps=1e-4)

arr, logS, _ = network.topLevelRepresentation()
q = np.log(arr) + logS
q /= (nX*nY)
print('J=',j,'lnZ (calculated)=',q,'lnZ (exact)=',exactIsing2D(j))
