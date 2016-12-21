import numpy as np
from scipy.integrate import quad  

from TNRG.Network.link import Link
from TNRG.Network.node import Node
from TNRG.Network.network import Network
from TNRG.TreeTensor.identityTensor import IdentityTensor
from TNRG.Tensor.arrayTensor import ArrayTensor

def PA2D(nX, nY, h, J, accuracy): 
	network = Network()
 
	# Place to store the tensors 
	lattice = [[] for i in range(nX)] 
	onSite = [[] for i in range(nX)] 
	bondV = [[] for i in range(nX)] 
	bondH = [[] for i in range(nX)] 
	bondL0 = [[] for i in range(nX)]
	bondL1 = [[] for i in range(nX)]
	bondL2 = [[] for i in range(nX)]
	bondL3 = [[] for i in range(nX)]
 
	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise. 
	for i in range(nX): 
		for j in range(nY): 
			lattice[i].append(Node(IdentityTensor(2, 17, accuracy=accuracy)))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively. 
	for i in range(nX): 
		for j in range(nY): 
			arr = np.zeros((2)) 
			arr[0] = np.exp(-h) 
			arr[1] = np.exp(h)
			onSite[i].append(Node(ArrayTensor(arr)))
 
	# Each bond term has two indices of width two and returns exp(-J*(1+delta(index0,index1))/2). 
	for i in range(nX): 
		for j in range(nY): 
			arr = np.zeros((2,2)) 
			arr[0][0] = np.exp(-J) 
			arr[1][1] = np.exp(-J) 
			arr[0][1] = np.exp(J) 
			arr[1][0] = np.exp(J) 
			bondV[i].append(Node(ArrayTensor(arr)))
			bondH[i].append(Node(ArrayTensor(arr)))

	# Add L-bonds
	for i in range(nX):
		for j in range(nY):
			arr = np.zeros((2,2,2))
			arr += 1./7
			arr[1,1,1] = 0
			bondL0[i].append(Node(ArrayTensor(np.copy(arr))))
			bondL1[i].append(Node(ArrayTensor(np.copy(arr))))
			bondL2[i].append(Node(ArrayTensor(np.copy(arr))))
			bondL3[i].append(Node(ArrayTensor(np.copy(arr))))
 
	# Attach links
	for i in range(nX): 
		for j in range(nY): 
			Link(lattice[i][j].buckets[0], onSite[i][j].buckets[0])

			Link(lattice[i][j].buckets[1], bondV[i][j].buckets[0])
			Link(lattice[i][j].buckets[2], bondV[i][(j+1)%nY].buckets[1])

			Link(lattice[i][j].buckets[3], bondH[i][j].buckets[0])
			Link(lattice[i][j].buckets[4], bondH[(i+1)%nX][j].buckets[1])

			Link(lattice[i][j].buckets[5], bondL0[i][j].buckets[0])
			Link(lattice[i][j].buckets[6], bondL0[i][(j+1)%nY].buckets[1])
			Link(lattice[i][j].buckets[7], bondL0[(i+1)%nX][j].buckets[2])

			Link(lattice[i][j].buckets[8], bondL1[i][j].buckets[0])
			Link(lattice[i][j].buckets[9], bondL1[i][(j-1)%nY].buckets[1])
			Link(lattice[i][j].buckets[10], bondL1[(i+1)%nX][j].buckets[2])

			Link(lattice[i][j].buckets[11], bondL2[i][j].buckets[0])
			Link(lattice[i][j].buckets[12], bondL2[i][(j+1)%nY].buckets[1])
			Link(lattice[i][j].buckets[13], bondL2[(i-1)%nX][j].buckets[2])

			Link(lattice[i][j].buckets[14], bondL3[i][j].buckets[0])
			Link(lattice[i][j].buckets[15], bondL3[i][(j-1)%nY].buckets[1])
			Link(lattice[i][j].buckets[16], bondL3[(i-1)%nX][j].buckets[2])



	# Add to Network
	for i in range(nX):
		for j in range(nY):
			network.addNode(lattice[i][j])
			network.addNode(onSite[i][j])
			network.addNode(bondV[i][j])
			network.addNode(bondH[i][j])
			network.addNode(bondL0[i][j])
			network.addNode(bondL1[i][j])
			network.addNode(bondL2[i][j])
			network.addNode(bondL3[i][j])
 
	return network