import sys
sys.path.append('../TensorNetwork/')
from network import Network
from latticeNode import latticeNode
import numpy as np
from scipy.integrate import quad
import cProfile

def IsingSolve(nX, nY, nZ, h, J):
	network = Network()

	# Place to store the tensors
	lattice = [[[] for j in range(nY)] for i in range(nX)]
	onSite = [[[] for j in range(nY)] for i in range(nX)]
	bondV = [[[] for j in range(nY)] for i in range(nX)]
	bondH = [[[] for j in range(nY)] for i in range(nX)]
	bondZ = [[[] for j in range(nY)] for i in range(nX)]
	bondL = [[[] for j in range(nY)] for i in range(nX)]

	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise.
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j].append(latticeNode(2,network))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively.
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				arr = np.zeros((2))
				arr[0] = np.exp(-h)
				arr[1] = np.exp(h)
				onSite[i][j].append(network.addNodeFromArray(arr))

	# Each bond term has two indices of width two and returns exp(-J*(1+delta(index0,index1))/2).
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				arr = np.zeros((2,2))
				arr[0][0] = np.exp(-J)
				arr[1][1] = np.exp(-J)
				arr[0][1] = np.exp(J)
				arr[1][0] = np.exp(J)
				bondV[i][j].append(network.addNodeFromArray(np.copy(arr)))
				bondH[i][j].append(network.addNodeFromArray(np.copy(arr)))
				bondZ[i][j].append(network.addNodeFromArray(np.copy(arr)))

	# Compute remaining bonds. Centre, -x, +x, -y, +y, -z, +z
	arr = np.ones((2,2,2,2,2,2,2))

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

	arr = np.random.rand(2,2,2,2,2,2,2)+0.5

	# Make L-bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
					bondL[i][j].append(network.addNodeFromArray(np.copy(arr)))


	# Attach on-site bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j][k].addLink(onSite[i][j][k],0)

	# Attach nearest-neighbor bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j][k].addLink(bondV[i][j][k],0)
				lattice[i][j][k].addLink(bondV[i][(j+1)%nY][k],1)
				lattice[i][j][k].addLink(bondH[i][j][k],0)
				lattice[i][j][k].addLink(bondH[(i+1)%nX][j][k],1)
				lattice[i][j][k].addLink(bondZ[i][j][k],0)
				lattice[i][j][k].addLink(bondZ[i][j][(k+1)%nZ],1)

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

	network.trace()

	counter = 1
	while len(network.topLevelLinks()) > 0:
		if counter%1 == 0:
			n = network.largestTopLevelTensor()
			t = n.tensor()

			print len(network.topLevelNodes()), network.topLevelSize(), np.product(t.shape()), t.shape()

		counter += 1

		network.merge(mergeL=True,compress=True)

	return np.log(list(network.topLevelNodes())[0].tensor().array()) + list(network.topLevelNodes())[0].logScalar()

#print IsingSolve(10,10,10,0,0.5)/1000
#print IsingSolve(10,10,0,0.5)/100
print IsingSolve(5,5,5,0.2,0.2)
#print cProfile.run('IsingSolve(20,20,20,0.0,1.0)/(20)**3')

