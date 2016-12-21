import numpy as np
from scipy.integrate import quad  

from TNRG.Network.link import Link
from TNRG.Network.node import Node
from TNRG.Network.network import Network
from TNRG.TreeTensor.identityTensor import IdentityTensor
from TNRG.Tensor.arrayTensor import ArrayTensor

import numpy as np
from scipy.integrate import quad  

from TNRG.Network.link import Link
from TNRG.Network.node import Node
from TNRG.Network.network import Network
from TNRG.TreeTensor.identityTensor import IdentityTensor
from TNRG.Tensor.arrayTensor import ArrayTensor

def PA2D(nX, nY, h, J, q, accuracy): 
	network = Network()
 
	# Place to store the tensors 
	lattice = [[] for i in range(nX)] 
	bondL = [[] for i in range(nX)]
 
	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise. 
	for i in range(nX): 
		for j in range(nY): 
			for k in range(nZ):
				lattice[i][j].append(Node(IdentityTensor(2, 5, accuracy=accuracy)))

	arr = np.zeros((2,2))

	# 2-point
	arr[0][0] = np.exp(-J)
	arr[1][1] = np.exp(-J)
	arr[0][1] = np.exp(J)
	arr[1][0] = np.exp(J)

	# 1-point
	arr[0] *= np.exp(h/4)
	arr[1] *= np.exp(-h/4)

	# Expand
	arr = np.einsum('ij,ik,il,ia->ijkla',arr,arr,arr,arr)

	# 3-point
	arr[1,1,:,1,:] = 0
	arr[1,1,:,:,1] = 0
	arr[1,:,1,1,:] = 0
	arr[1,:,1,:,1] = 0

	# 4-point
	arr[1,1,1,1,:] = np.exp(q)
	arr[1,1,1,:,1] = np.exp(q)
	arr[1,1,:,1,1] = np.exp(q)
	arr[1,:,1,1,1] = np.exp(q)

	# 5-point
	for j in range(2):
		for k in range(2):
			for l in range(2):
				for m in range(2):
					if j + k + l + m >= 4:
							arr[1,j,k,l,m] = 0

	# Make L-bonds
	for i in range(nX):
		for j in range(nY):
				bondL[i].append(Node(ArrayTensor(arr)))
 
	# Attach links
	for i in range(nX): 
		for j in range(nY): 
			Link(lattice[i][j].buckets[0], bondL[i][j].buckets[0])
			Link(lattice[i][j].buckets[1], bondL[(i+1)%nX][j].buckets[1])
			Link(lattice[i][j].buckets[2], bondL[i-1][j].buckets[2])
			Link(lattice[i][j].buckets[3], bondL[i][(j+1)%nY].buckets[3])
			Link(lattice[i][j].buckets[4], bondL[i][j-1].buckets[4])

	# Add to Network
	for i in range(nX):
		for j in range(nY):
			network.addNode(lattice[i][j])
			network.addNode(bondL[i][j])
 
	return network