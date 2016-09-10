import sys 
sys.path.append('../TensorNetwork/') 
from networkTree import NetworkTree
from latticeNode import latticeNode 
import numpy as np 
from scipy.integrate import quad 
import cProfile 
 
def IsingModel2D(nX, nY, h, J): 
	network = NetworkTree() 
 
	# Place to store the tensors 
	lattice = [[] for i in range(nX)] 
	onSite = [[] for i in range(nX)] 
	bondV = [[] for i in range(nX)] 
	bondH = [[] for i in range(nX)] 
 
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
 
	return network, lattice 
 
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
 

def exactIsing2D(J): 
	k = 1/np.sinh(2*J)**2 
	def f(x): 
		return np.log(np.cosh(2*J)**2 + (1/k)*np.sqrt(1+k**2-2*k*np.cos(2*x))) 
	inte = quad(f,0,np.pi)[0] 
 
	return np.log(2)/2 + (1/(2*np.pi))*inte 

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

