import numpy as np
from scipy.integrate import quad 
from copy import deepcopy

from TNRG.Network.link import Link
from TNRG.Network.node import Node
from TNRG.Network.network import Network
from TNRG.TreeTensor.identityTensor import IdentityTensor
from TNRG.TreeTensor.treeTensor import TreeTensor
from TNRG.Tensor.arrayTensor import ArrayTensor

def PA3D(nX, nY, nZ, h, J, q, accuracy): 
	network = Network()
 
	# Place to store the tensors 
	lattice = [[[] for j in range(nY)] for i in range(nX)] 
	bondL = [[[] for j in range(nY)] for i in range(nX)]
 
	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise. 
	for i in range(nX): 
		for j in range(nY): 
			for k in range(nZ):
				lattice[i][j].append(Node(IdentityTensor(2, 7, accuracy=accuracy)))

	arr = np.zeros((2,2))

	# 2-point
	arr[0][0] = np.exp(-J)
	arr[1][1] = np.exp(-J)
	arr[0][1] = np.exp(J)
	arr[1][0] = np.exp(J)

	# 1-point
	arr[0] *= np.exp(h/6)
	arr[1] *= np.exp(-h/6)

	# Expand
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
	arr[1,1,1,1,:,:,:] = np.exp(q)
	arr[1,1,1,:,1,:,:] = np.exp(q)
	arr[1,1,1,:,:,1,:] = np.exp(q)
	arr[1,1,1,:,:,:,1] = np.exp(q)
	arr[1,1,:,1,1,:,:] = np.exp(q)
	arr[1,:,1,1,1,:,:] = np.exp(q)
	arr[1,:,:,1,1,1,:] = np.exp(q)
	arr[1,:,:,1,1,:,1] = np.exp(q)
	arr[1,1,:,:,:,1,1] = np.exp(q)
	arr[1,:,1,:,:,1,1] = np.exp(q)
	arr[1,:,:,1,:,1,1] = np.exp(q)
	arr[1,:,:,:,1,1,1] = np.exp(q)


	# 5-point
	for j in range(2):
		for k in range(2):
			for l in range(2):
				for m in range(2):
					for n in range(2):
						for p in range(2):
							if j + k + l + m + n + p >= 4:
									arr[1,j,k,l,m,n,p] = 0

	t = ArrayTensor(arr)
	tt = TreeTensor(accuracy)
	tt.addTensor(t)

	# Make L-bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				bondL[i][j].append(Node(deepcopy(tt)))
 
	# Attach links
	for i in range(nX): 
		for j in range(nY): 
			for k in range(nZ):
				Link(lattice[i][j][k].buckets[0], bondL[i][j][k].buckets[0])
				Link(lattice[i][j][k].buckets[1], bondL[(i+1)%nX][j][k].buckets[1])
				Link(lattice[i][j][k].buckets[2], bondL[i-1][j][k].buckets[2])
				Link(lattice[i][j][k].buckets[3], bondL[i][(j+1)%nY][k].buckets[3])
				Link(lattice[i][j][k].buckets[4], bondL[i][j-1][k].buckets[4])
				Link(lattice[i][j][k].buckets[5], bondL[i][j][(k+1)%nZ].buckets[5])
				Link(lattice[i][j][k].buckets[6], bondL[i][j][k-1].buckets[6])

	# Add to Network
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				network.addNode(lattice[i][j][k])
				network.addNode(bondL[i][j][k])
 
	return network