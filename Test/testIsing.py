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

	network.contract(mergeL=True, compressL=True, eps=1e-4)

	print(list(network.topLevelNodes())[0].logScalar()/(nX*nY))
	print(exactIsing(J))

	network.optimize(mergeL=True, compressL=True, eps=1e-4)

	print(list(network.topLevelNodes())[0].logScalar()/(nX*nY))
	print(exactIsing(J))

	exit()

	nn, arr, bucketList = network.view(set([lattice[0][0],lattice[0][1],lattice[0][2]]))

	print(arr)
	print(arr.shape)

	lattice[0][0].addDim()
	lattice[0][1].addDim()

	network.contract(mergeL=True, compressL=True, eps=1e-4)

	return np.exp(list(network.topLevelNodes())[0].logScalar())*list(network.topLevelNodes())[0].tensor().array()

#	return np.log(list(network.topLevelNodes())[0].tensor().array()) + list(network.topLevelNodes())[0].logScalar()

def exactIsing(J):
	k = 1/np.sinh(2*J)**2
	def f(x):
		return np.log(np.cosh(2*J)**2 + (1/k)*np.sqrt(1+k**2-2*k*np.cos(2*x)))
	inte = quad(f,0,np.pi)[0]

	return np.log(2)/2 + (1/(2*np.pi))*inte



#print IsingSolve(20,20,0,0.5)/400,exactIsing(0.5)
corr = IsingSolve(10,10,0,0.5)

print(corr/np.sum(corr))

#print cProfile.run('print IsingSolve(10,10,0,0.5)/100')

exit()

print(IsingSolve(7,7,4.0,0)/49,np.log(np.exp(4) + np.exp(-4)))

for j in np.linspace(-1,1,num=10):
	q = IsingSolve(10,10,0,j)/100
	print(j,q,exactIsing(j))

