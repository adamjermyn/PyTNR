import tensors
import numpy as np
import networkx

def IsingRandom(n, nBonds):
	network = tensors.TensorNetwork()

	# Place to store the tensors
	lattice = []
	bonds = [[] for i in range(n*nBonds)]

	# Generate bonds
	bondInd1 = np.random.randint(0,n,size=(n,nBonds))
	bondInd2 = np.random.randint(0,n-1,size=(n,nBonds))

	numBonds = []
	for i in range(n):
		numBonds.append()

	for i in range(n):
		arr = np.zeros((2 for i in nBonds))
		np.fill_diagonal(arr,1.0)
		lattice.append(network.addTensor(np.copy(arr)))

	for i in range(n):
		for j in range(nBonds):
			arr = np.zeros((2,2))
			m = np.random.randn()
			arr[0][0] = np.exp(-m)
			arr[1][1] = np.exp(-m)
			arr[0][1] = np.exp(m)
			arr[1][0] = np.exp(m)
			bonds[i].append(network.addTensor(np.copy(arr)))

	for i in range(n):
		for j in range(nBonds):
			lattice[i].addLink()