import sys
sys.path.append('../TensorNetwork/')
sys.path.append('../Models/')
import numpy as np
from isingModel import IsingModel2D, exactIsing2D, IsingModel3D, RandomIsingModel2D
import cProfile

#######
# 2D ##
#######

print('Testing 2D Regular Ising Model - J sweep, h=0:')

nX = 10
nY = 10

for j in np.linspace(-1,1,num=10):
	network, lattice = IsingModel2D(nX, nY, 0, j)
	network.contract(mergeL=True, compressL=True, eps=1e-4)
	arr, logS, _ = network.topLevelRepresentation()
	q = np.log(arr) + logS
	q /= (nX*nY)
	print('J=',j,'lnZ (calculated)=',q,'lnZ (exact)=',exactIsing2D(j))

#######
# 2D ##
#######

print('Testing 2D Random Ising Model:')

nX = 10
nY = 10

network, lattice = RandomIsingModel2D(nX, nY)
network.contract(mergeL=True, compressL=True, eps=1e-4)
arr, logS, _ = network.topLevelRepresentation()
q = np.log(arr) + logS
q /= (nX*nY)
print('lnZ (calculated)=',q)

#######
# 3D ##
#######

print('Testing 3D Regular Ising Model - J=0.5,h=0.5:')

nX = 8
nY = 8
nZ = 8

network, lattice=  IsingModel3D(nX, nY, nZ, 0.5, 0.5)
network.contract(mergeL=True, compressL=True, eps=1e-4)
arr, logS, _ = network.topLevelRepresentation()
q = np.log(arr) + logS
print('lnZ (calculated)=',q)