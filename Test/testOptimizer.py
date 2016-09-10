import sys
sys.path.append('../TensorNetwork/')
sys.path.append('../Models/')
import numpy as np
from isingModel import IsingModel2D, exactIsing2D, IsingModel3D, RandomIsingModel2D
import cProfile

nX = 4
nY = 4

network, lattice = IsingModel2D(nX, nY, 0.0, 0.5)
network.contract(mergeL=True, compressL=True, eps=1e-4)
arra, logS, _ = network.topLevelRepresentation()
qq = logS
qq /= (nX*nY)

network.optimize(mergeL=True, compressL=True, eps=1e-4)

arr, logS, _ = network.topLevelRepresentation()
print(arra)
print(arr)
q = logS
q /= (nX*nY)
print('lnZ (calculated, optimized)=',q,)
print('lnZ (calculated)=',qq)
print('lnZ (exact)=',exactIsing2D(0.5))
