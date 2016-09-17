import sys 
sys.path.append('../TensorNetwork/') 
sys.path.append('../Models/') 
from networkTree import NetworkTree
from latticeNode import latticeNode 
import numpy as np 
from scipy.integrate import quad 
import cProfile 
from isingModel import exactIsing2D
from tileTree import makeTileTreeFromArrays, linkTrees


def isingModel2Dperiodic(h,j):
	latticeLength = 2
	dimensions = [2,2]

	bondArrays = []
	bondArrays.append(np.array([np.exp(-h),np.exp(h)]))
	bondArrays.append(np.array([[np.exp(-j),np.exp(j)],[np.exp(j),np.exp(-j)]]))
	bondArrays.append(np.array([[np.exp(-j),np.exp(j)],[np.exp(j),np.exp(-j)]]))

	footprints = []
	footprints.append([[0,0]])
	footprints.append([[0,0],[0,1]])
	footprints.append([[0,0],[1,0]])

	nt, periodicLinks = makeTileTreeFromArrays(latticeLength, dimensions, bondArrays, footprints)

	nt.contract(mergeL=True, compressL=True, eps=1e-10)

	arr, logS, _ = nt.topLevelRepresentation()
	print(logS)

	nt, periodicLinks = linkTrees(0, nt, periodicLinks)

	nt.contract(mergeL=True, compressL=True, eps=1e-10)

	arr, logS, _ = nt.topLevelRepresentation()
	print(logS)

print('----')
isingModel2Dperiodic(3.0,0)
print('----')
isingModel2Dperiodic(0.0,0.3333)