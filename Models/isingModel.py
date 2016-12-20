import numpy as np
from scipy.integrate import quad  

from TNRG.Network.link import Link
from TNRG.Network.node import Node
from TNRG.Network.network import Network
from TNRG.TreeTensor.identityTensor import IdentityTensor
from TNRG.Tensor.arrayTensor import ArrayTensor

def IsingModel1D(nX, h, J, accuracy): 
	network = Network()
 
	# Place to store the tensors 
	lattice = [] 
	onSite = []
	bond = []
 
	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise. 
	for i in range(nX): 
		lattice.append(Node(IdentityTensor(2, 3, accuracy=accuracy)))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively. 
	for i in range(nX): 
		arr = np.zeros((2)) 
		arr[0] = np.exp(-h) 
		arr[1] = np.exp(h)
		onSite.append(Node(ArrayTensor(arr)))
 
	# Each bond term has two indices of width two and returns exp(-J*(1+delta(index0,index1))/2). 
	for i in range(nX): 
		arr = np.zeros((2,2)) 
		arr[0][0] = np.exp(-J) 
		arr[1][1] = np.exp(-J) 
		arr[0][1] = np.exp(J) 
		arr[1][0] = np.exp(J) 
		bond.append(Node(ArrayTensor(arr)))
 
	# Attach links
	for i in range(nX): 
		Link(lattice[i].buckets[0], onSite[i].buckets[0])
		Link(lattice[i].buckets[1], bond[i].buckets[0])
		Link(lattice[i].buckets[2], bond[(i+1)%nX].buckets[1])

	# Add to Network
	for i in range(nX):
		network.addNode(lattice[i])
		network.addNode(onSite[i])
		network.addNode(bond[i])
 
	return network

def exactIsing1Dh(h):
	return np.log(2*np.cosh(h))

def exactIsing1DJ(n, J):
	J = -J
	l1 = 2*np.cosh(J)
	l2 = 2*np.sinh(J)

	q = l2/l1

	f = 0
	if abs(q)**n < 1e-10:
		f = np.log1p(q**n)/n
	else:
		f = np.log1p((l2/l1)**n)/n

	return np.log(l1) + f

def IsingModel2D(nX, nY, h, J, accuracy): 
	network = Network()
 
	# Place to store the tensors 
	lattice = [[] for i in range(nX)] 
	onSite = [[] for i in range(nX)] 
	bondV = [[] for i in range(nX)] 
	bondH = [[] for i in range(nX)] 
 
	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise. 
	for i in range(nX): 
		for j in range(nY): 
			lattice[i].append(Node(IdentityTensor(2, 5, accuracy=accuracy)))

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
 
	# Attach links
	for i in range(nX): 
		for j in range(nY): 
			Link(lattice[i][j].buckets[0], onSite[i][j].buckets[0])
			Link(lattice[i][j].buckets[1], bondV[i][j].buckets[0])
			Link(lattice[i][j].buckets[2], bondV[i][(j+1)%nY].buckets[1])
			Link(lattice[i][j].buckets[3], bondH[i][j].buckets[0])
			Link(lattice[i][j].buckets[4], bondH[(i+1)%nX][j].buckets[1])

	# Add to Network
	for i in range(nX):
		for j in range(nY):
			network.addNode(lattice[i][j])
			network.addNode(onSite[i][j])
			network.addNode(bondV[i][j])
			network.addNode(bondH[i][j])
 
	return network

def exactIsing2D(J): 
	k = 1/np.sinh(2*J)**2 
	def f(x): 
		return np.log(np.cosh(2*J)**2 + (1/k)*np.sqrt(1+k**2-2*k*np.cos(2*x))) 
	inte = quad(f,0,np.pi)[0] 
 
	return np.log(2)/2 + (1/(2*np.pi))*inte

''' 
def RandomIsingModel2D(nX, nY): 
	network = NetworkTree() 
 
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
			h = np.random.randn()
			arr[0] = np.exp(-h) 
			arr[1] = np.exp(h) 
			onSite[i].append(network.addNodeFromArray(arr)) 
			lattice[i][j].addLink(onSite[i][j],0) 
 
	# Each bond term has two indices of width two and returns exp(-J*(1+delta(index0,index1))/2). 
	for i in range(nX): 
		for j in range(nY): 
			arr = np.zeros((2,2)) 
			J = np.random.randn()
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
 
	return network, lattice 
 



def IsingModel3D(nX, nY, nZ, h, J):
	network = NetworkTree()

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

	return network, lattice
'''
