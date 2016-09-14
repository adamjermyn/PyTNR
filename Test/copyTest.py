import sys
sys.path.append('../TensorNetwork/')
sys.path.append('../Models/')
import numpy as np
from isingModel import IsingModel2D, exactIsing2D, IsingModel3D, RandomIsingModel2D
import cProfile
from copy import deepcopy
sys.setrecursionlimit(10000)
nX = 4
nY = 4

network, lattice = IsingModel2D(nX, nY, 0.0, 0.5)
network.contract(mergeL=True, compressL=True, eps=1e-4)
arra, logS, _ = network.topLevelRepresentation()
qq = logS
print(qq)

network2 = deepcopy(network)

arra, logS, _ = network.topLevelRepresentation()
print(qq)

print(network.topLevelNodes())
print(network2.topLevelNodes())