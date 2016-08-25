import sys
sys.path.append('../TensorNetwork/')
from network import Network
from latticeNode import latticeNode
import numpy as np
from scipy.integrate import quad
import cProfile
from compress import compress

def IsingSolve(nX, nY, nZ, h, J):
	network = Network()

	# Place to store the tensors
	lattice = [[[] for j in range(nY)] for i in range(nX)]
	bondL = [[[] for j in range(nY)] for i in range(nX)]

	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise.
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j].append(latticeNode(2,network))

	# Compute remaining bonds. Centre, -x, +x, -y, +y, -z, +z
	arr = np.zeros((2,2))
	arr[0][0] = np.exp(-J)
	arr[1][1] = np.exp(-J)
	arr[0][1] = np.exp(J)
	arr[1][0] = np.exp(J)
	arr[0] *= np.exp(-h/6)
	arr[1] *= np.exp(h/6)
	arr = np.einsum('ij,ik,il,ia,ib,ic->ijklabc',arr,arr,arr,arr,arr,arr)

	# 3-point
	arr[1,1,:,1,:,:,:] = 0
	arr[1,1,:,:,1,:,:] = 0
	arr[1,1,:,:,:,1,:] = 0
	arr[1,1,:,:,:,:,1] = 0
	arr[1,:,1,1,:,:,:] = 0
	arr[1,:,1,:,1,:,:] = 0
	arr[1,:,1,:,:,1,:] = 0
	arr[1,:,1,:,:,:,1] = 0

	arr[1,1,:,1,:,:,:] = 0
	arr[1,:,1,1,:,:,:] = 0
	arr[1,:,:,1,:,1,:] = 0
	arr[1,:,:,1,:,:,1] = 0
	arr[1,1,:,:,1,:,:] = 0
	arr[1,:,1,:,1,:,:] = 0
	arr[1,:,:,:,1,1,:] = 0
	arr[1,:,:,:,1,:,1] = 0

	arr[1,1,:,:,:,1,:] = 0
	arr[1,:,1,:,:,1,:] = 0
	arr[1,:,:,1,:,1,:] = 0
	arr[1,:,:,:,1,1,:] = 0
	arr[1,1,:,:,:,:,1] = 0
	arr[1,:,1,:,:,:,1] = 0
	arr[1,:,:,1,:,:,1] = 0
	arr[1,:,:,:,1,:,1] = 0

	# 4-point
	arr[1,1,1,1,:,:,:] = 0.1
	arr[1,1,1,:,1,:,:] = 0.1
	arr[1,1,1,:,:,1,:] = 0.1
	arr[1,1,1,:,:,:,1] = 0.1
	arr[1,1,:,1,1,:,:] = 0.1
	arr[1,:,1,1,1,:,:] = 0.1
	arr[1,:,:,1,1,1,:] = 0.1
	arr[1,:,:,1,1,:,1] = 0.1
	arr[1,1,:,:,:,1,1] = 0.1
	arr[1,:,1,:,:,1,1] = 0.1
	arr[1,:,:,1,:,1,1] = 0.1
	arr[1,:,:,:,1,1,1] = 0.1

	# Make L-bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
					bondL[i][j].append(network.addNodeFromArray(np.copy(arr)))

	# Attach L-bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j][k].addLink(bondL[i][j][k],0)
				lattice[i][j][k].addLink(bondL[(i+1)%nX][j][k],1)
				lattice[i][j][k].addLink(bondL[i-1][j][k],2)
				lattice[i][j][k].addLink(bondL[i][(j+1)%nY][k],3)
				lattice[i][j][k].addLink(bondL[i][j-1][k],4)
				lattice[i][j][k].addLink(bondL[i][j][(k+1)%nZ],5)
				lattice[i][j][k].addLink(bondL[i][j][k-1],6)

	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j][k] = lattice[i][j][k].merge(bondL[i][j][k], mergeL=False, compress=False)

	network.linkMerge(compress=False)

	print network

	counter = 0
	while len(network.topLevelLinks()) > 0:
		if counter%1 == 0:
			n = network.largestTopLevelTensor()
			t = n.tensor()

			print len(network.topLevelNodes()), network.topLevelSize(), np.product(t.shape()), t.shape()
			print network

		counter += 1

		network.merge(mergeL=True,compress=True)

	return np.log(list(network.topLevelNodes())[0].tensor().array()) + list(network.topLevelNodes())[0].logScalar()

#print IsingSolve(10,10,10,0,0.5)/1000
#print IsingSolve(10,10,0,0.5)/100
print IsingSolve(5,5,5,0.2,0.2)
#print cProfile.run('IsingSolve(20,20,20,0.0,1.0)/(20)**3')

