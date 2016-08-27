import sys
sys.path.append('../TensorNetwork/')
from network import Network
from latticeNode import latticeNode
import numpy as np
from scipy.integrate import quad
import cProfile

def IsingSolve(nX, nY, h, J):
	network = Network()

	# Place to store the tensors
	lattice = [[] for i in range(nX)]
	onSite = [[] for i in range(nX)]
	bondV = [[] for i in range(nX)]
	bondH = [[] for i in range(nY)]

	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise.
	for i in range(nX):
		for j in range(nY):
			lattice[i].append(latticeNode(2,network))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively.
	for i in range(nX):
		for j in range(nY):
			arr = np.zeros((2))
			arr[0] = np.exp(-h)
			arr[1] = np.exp(h)
			onSite[i].append(network.addNodeFromArray(arr))
			lattice[i][j].addLink(onSite[i][j],0)

	# Each bond term has two indices of width two and returns exp(-J*(1+delta(index0,index1))/2).
	for i in range(nX):
		for j in range(nY):
			arr = np.zeros((2,2))
			arr[0][0] = np.exp(-J)
			arr[1][1] = np.exp(-J)
			arr[0][1] = np.exp(J)
			arr[1][0] = np.exp(J)
			bondV[i].append(network.addNodeFromArray(np.copy(arr)))
			bondH[i].append(network.addNodeFromArray(np.copy(arr)))

	# Attach bond terms
	for i in range(nX):
		for j in range(nY):
			lattice[i][j].addLink(bondV[i][j],0)
			lattice[i][j].addLink(bondV[i][(j+1)%nY],1)
			lattice[i][j].addLink(bondH[i][j],0)
			lattice[i][j].addLink(bondH[(i+1)%nX][j],1)

	network.trace()

	counter = 0
	while len(network.topLevelLinks()) > 0:
		network.merge(mergeL=True,compress=True)

		if counter%20 == 0:
			t = network.largestTopLevelTensor()
			print len(network.topLevelNodes()),network.topLevelSize(), t.tensor().shape()
		counter += 1

	return lattice, network

def correlator(lattice, network, i,j,k,l):
	lattice[i][j].addDim()
	lattice[k][l].addDim()

	counter = 0

	while len(network.topLevelLinks()) > 0:
		network.merge(mergeL=True,compress=True)

		if counter%1 == 0:
			print len(network.topLevelNodes()),network.topLevelSize(), network.largestTopLevelTensor()
		counter += 1

	topNodes = list(network.topLevelNodes())
	topLevel = [np.exp(n.logScalar())*n.tensor().array() for n in topNodes]

	ret = topLevel[0]
	for q in range(1,len(topLevel)):
		ret = np.tensordot(ret, topLevel[q],axes=0)

	lattice[i][j].removeDim()
	lattice[k][l].removeDim()	

	print i,j,k,l

	return ret

nX = 10
nY = 10

lattice, network = IsingSolve(nX,nY,0,-0.4)

data = np.zeros((nX,nY))

for i in range(nX):
	for j in range(nY):
		if i==0 and j==0:
			data[i,j] = 1.0
		else:
			corr = correlator(lattice, network, i, j, 0, 0)
			corr /= np.sum(corr)
			data[i,j] = corr[0][0]+corr[1][1]-corr[0][1]-corr[1][0]


import matplotlib.pyplot as plt

plt.imshow(np.log(np.abs(data)))
plt.show()

