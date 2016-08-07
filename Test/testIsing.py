import tensors
import numpy as np
import networkx
from scipy.integrate import quad
import matplotlib.pyplot as plt

def IsingSolve(nX, nY, h, J):
	network = tensors.TensorNetwork()

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
			lattice[i].append(network.addTensor(arr))

	# Each on-site term has one index of width two, and returns exp(-h) or exp(h) for 0 or 1 respectively.
	for i in range(nX):
		for j in range(nY):
			arr = np.zeros((2))
			arr[0] = np.exp(-h)
			arr[1] = np.exp(h)
			onSite[i].append(network.addTensor(arr))
			lattice[i][j].addLink(onSite[i][j],0,0,kind='outside')

	# Each bond term has two indices of width two and returns exp(-J*(1+delta(index0,index1))/2).
	for i in range(nX):
		for j in range(nY):
			arr = np.zeros((2,2))
			arr[0][0] = np.exp(-J)
			arr[1][1] = np.exp(-J)
			arr[0][1] = np.exp(J)
			arr[1][0] = np.exp(J)
			bondV[i].append(network.addTensor(np.copy(arr)))
			bondH[i].append(network.addTensor(np.copy(arr)))

	# Attach bond terms
	for i in range(nX):
		for j in range(nY):
			lattice[i][j].addLink(bondV[i][j],1,0,kind='outside')
			lattice[i][j].addLink(bondV[i][(j+1)%nY],2,1,kind='outside')
			lattice[i][j].addLink(bondH[i][j],3,0,kind='outside')
			lattice[i][j].addLink(bondH[(i+1)%nX][j],4,1,kind='outside')

	# Print structure
#	for i in range(nX):
#		for j in range(nY):
#			print lattice[i][j]


	# Clean up links
	for t in network.tensors:
		t.mergeAllLinks()

	# Compute partition function
	while len(network.tensors) > 1:
#		print '----'
#		print np.sum([t.array.size for t in network.tensors])
		shapes = [t.array.shape for t in network.tensors]
		sizes = np.array([t.array.size for t in network.tensors])
		print len(network.tensors),np.sum(sizes),np.min(sizes),np.max(sizes),shapes[np.argmin(sizes)],shapes[np.argmax(sizes)]
#		for t in network.tensors:
#			print(t.array.shape)
		network = network.merge()
		netRed = network.compress()
#		network = network.split()
#		print '----'

#		print [t.array.size for t in network.tensors]

	return network.logScalar

def exactIsing(J):
	k = 1/np.sinh(2*J)**2
	def f(x):
		return np.log(np.cosh(2*J)**2 + (1/k)*np.sqrt(1+k**2-2*k*np.cos(2*x)))
	inte = quad(f,0,np.pi)[0]

	return np.log(2)/2 + (1/(2*np.pi))*inte

#print IsingSolve(7,7,2.0,0)/49,np.log(np.exp(2) + np.exp(-2))

#print IsingSolve(7,7,4.0,0)/49,np.log(np.exp(4) + np.exp(-4))

for j in np.linspace(-1,1,num=10):
	q = IsingSolve(20,20,0,j)/400
	print(j,q,exactIsing(j))

