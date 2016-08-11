import sys
sys.path.append('../TensorNetwork/')
from network import Network
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
			arr = np.zeros((2,2,2,2,2))
			np.fill_diagonal(arr,1.0)
			lattice[i].append(network.addNodeFromArray(arr))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively.
	for i in range(nX):
		for j in range(nY):
			arr = np.zeros((2))
			arr[0] = np.exp(-h)
			arr[1] = np.exp(h)
			onSite[i].append(network.addNodeFromArray(arr))
			lattice[i][j].addLink(onSite[i][j],0,0)

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
			lattice[i][j].addLink(bondV[i][j],1,0)
			lattice[i][j].addLink(bondV[i][(j+1)%nY],2,1)
			lattice[i][j].addLink(bondH[i][j],3,0)
			lattice[i][j].addLink(bondH[(i+1)%nX][j],4,1)


	while len(network.topLevelLinks()) > 0:
		network.merge()
		network.linkMerge(compress=True)
		network.trace()

		print network.topLevelSize(), network.largestTopLevelTensor()

	return np.log(list(network.topLevelNodes())[0].tensor().array())

def exactIsing(J):
	k = 1/np.sinh(2*J)**2
	def f(x):
		return np.log(np.cosh(2*J)**2 + (1/k)*np.sqrt(1+k**2-2*k*np.cos(2*x)))
	inte = quad(f,0,np.pi)[0]

	return np.log(2)/2 + (1/(2*np.pi))*inte

#print cProfile.run('IsingSolve(7,7,2.0,0)/49,np.log(np.exp(2) + np.exp(-2))')

print IsingSolve(7,7,2.0,0)/49,np.log(np.exp(2) + np.exp(-2))

exit()

print IsingSolve(7,7,4.0,0)/49,np.log(np.exp(4) + np.exp(-4))

for j in np.linspace(-1,1,num=10):
	q = IsingSolve(10,10,0,j)/100
	print(j,q,exactIsing(j))

