import sys
sys.path.append('../TensorNetwork/')
from network import Network
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


	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise.
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				arr = np.zeros((2,2,2,2,2,2,2))
				np.fill_diagonal(arr,1.0)
				lattice[i][j].append(network.addNodeFromArray(arr))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively.
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				arr = np.zeros((2))
				arr[0] = np.exp(-h)
				arr[1] = np.exp(h)
				onSite[i][j].append(network.addNodeFromArray(arr))
				lattice[i][j][k].addLink(onSite[i][j][k],0,0)

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

	# Attach bond terms
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j][k].addLink(bondV[i][j][k],1,0)
				lattice[i][j][k].addLink(bondV[i][(j+1)%nY][k],2,1)
				lattice[i][j][k].addLink(bondH[i][j][k],3,0)
				lattice[i][j][k].addLink(bondH[(i+1)%nX][j][k],4,1)
				lattice[i][j][k].addLink(bondZ[i][j][k],5,0)
				lattice[i][j][k].addLink(bondZ[i][j][(k+1)%nZ],6,1)

	network.trace()

	counter = 0
	while len(network.topLevelLinks()) > 0:
		network.merge(mergeL=True,compress=True)

		if counter%20 == 0:
			print len(network.topLevelNodes()), network.topLevelSize(), network.largestTopLevelTensor()
		counter += 1

	return np.log(list(network.topLevelNodes())[0].tensor().array()) + list(network.topLevelNodes())[0].logScalar()

#print IsingSolve(10,10,10,0,0.5)/1000
#print IsingSolve(10,10,0,0.5)/100
IsingSolve(20,20,20,0.0,1.0)/(20)**3
#print cProfile.run('IsingSolve(20,20,20,0.0,1.0)/(20)**3')

