import sys 
sys.path.append('../TensorNetwork/') 
sys.path.append('../Models/') 
from networkTree import NetworkTree
from periodicNetworkTree import PeriodicNetworkTree
from latticeNode import latticeNode 
import numpy as np 
from scipy.integrate import quad 
import cProfile 
from isingModel import exactIsing2D

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

	p = PeriodicNetworkTree(latticeLength, dimensions, bondArrays, footprints)

	for _ in range(4):
		p.contract(mergeL=True,compressL=True,eps=1e-10)
		p.expand(0)
		p.contract(mergeL=True,compressL=True,eps=1e-10)
		p.expand(1)
	p.contract(mergeL=True,compressL=True,eps=1e-10)

	dims = p.dimensions

	pp, _, _, _, _ = p.copySubset(set(p.topLevelNodes()))

	pp.contract(mergeL=True, compressL=True, eps=1e-10)

	arr, logS, _ = pp.topLevelRepresentation()
	print(logS/(np.product(dims)))

	if h == 0:
		print(exactIsing2D(j))
	elif j == 0:
		print(np.log((np.exp(-h)+np.exp(h))))

isingModel2Dperiodic(3.0,0)
isingModel2Dperiodic(0.0,0.3333)