import sys
sys.path.append('../TensorNetwork/')
from networkTree import NetworkTree
from latticeNode import latticeNode
import numpy as np
from scipy.integrate import quad
import cProfile
from compress import compress

def makeTop(node):
	n = node
	while n.parent() is not None:
		n = n.parent()
	return n


def otherIsingSolve(nX, nY, nZ, h, J, q):
	# TODO: The logic here should probably be standardised to make
	# solving lattices systematically easier. It should also be parallelised. 

	network = NetworkTree()

	# Place to store the tensors
	lattice = [[[] for j in range(nY)] for i in range(nX)]
	bondL = [[[] for j in range(nY)] for i in range(nX)]

	# Each lattice site has seven indices of width five, and returns zero if they are unequal and one otherwise.
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
				lattice[i][j].append(latticeNode(2,network))

	# Compute remaining bonds. Centre, -x, +x, -y, +y, -z, +z

	# 2-point
	arr = np.zeros((2,2))
	arr[0][0] = np.exp(-J)
	arr[1][1] = np.exp(-J)
	arr[0][1] = np.exp(J)
	arr[1][0] = np.exp(J)
	arr[0] *= np.exp(h/6)
	arr[1] *= np.exp(-h/6)
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

	# Make L-bonds
	for i in range(nX):
		for j in range(nY):
			for k in range(nZ):
					bondL[i][j].append(network.addNodeFromArray(np.copy(arr)))

	# Attach L-bonds
	for i in range(nX):
		for j in range(nY):
			print(i,j)
			for k in range(nZ):
				lattice[i][j][k].addLink(bondL[i][j][k],0)
				lattice[i][j][k].addLink(bondL[(i+1)%nX][j][k],1)
				lattice[i][j][k].addLink(bondL[i-1][j][k],2)
				lattice[i][j][k].addLink(bondL[i][(j+1)%nY][k],3)
				lattice[i][j][k].addLink(bondL[i][j-1][k],4)
				lattice[i][j][k].addLink(bondL[i][j][(k+1)%nZ],5)
				lattice[i][j][k].addLink(bondL[i][j][k-1],6)

	for i in range(nX):
		for j in range(nY):
			print(i,j)
			for k in range(nZ):
				lattice[i][j][k] = lattice[i][j][k].merge(bondL[i][j][k], mergeL=True, compressL=True,eps=0.3)

	nnX = nX
	nnY = nY
	nnZ = nZ
	stop = False
	counter = 0
	while len(network.topLevelLinks()) > 0:
		if nnX == 1 and nnY == 1 and nnZ == 1 or stop:
			break

		print(len(network.topLevelNodes()))

		network.compressLinks()
		print(network)

		if nnX > 1:
			for ii in range(nnX//2):
				for j in range(nnY):
					for k in range(nnZ):
						i = ii*2+(counter%2)
						for q in range(3):
							for p in range(3):
								for t in range(3):
									lattice[(i+q-1)%nnX][(j+p-1)%nnY][(k+t-1)%nnZ] = makeTop(lattice[(i+q-1)%nnX][(j+p-1)%nnY][(k+t-1)%nnZ])
						if lattice[i][j][k] in lattice[(i+1)%nnX][j][k].connectedHigh():
							lattice[i][j][k].merge(lattice[(i+1)%nnX][j][k],compressL=True,mergeL=True,eps=1e-4)
						else:
							stop = True
						print(i,j,k,nnX,nnY,nnZ,lattice[i][j][k].tensor().shape())

			lattice = lattice[::2]
			nnX = len(lattice)
			for i in range(nnX):
				for j in range(nnY):
					for k in range(nnZ):
							makeTop(lattice[i][j][k])
		print(network)
		network.compressLinks()
		print(network)

		if nnX == 1 and nnY == 1 and nnZ == 1 or stop:
			break

		print(len(network.topLevelNodes()))

		if nnY > 1:
			for jj in range(nnY//2):
				for i in range(nnX):
					for k in range(nnZ):
						j = jj*2+(counter%2)
						for q in range(3):
							for p in range(3):
								for t in range(3):
									lattice[(i+q-1)%nnX][(j+p-1)%nnY][(k+t-1)%nnZ] = makeTop(lattice[(i+q-1)%nnX][(j+p-1)%nnY][(k+t-1)%nnZ])
						if lattice[i][j][k] in lattice[i][(j+1)%nnY][k].connectedHigh():
							lattice[i][j][k].merge(lattice[i][(j+1)%nnY][k],compressL=True,mergeL=True,eps=1e-4)
						else:
							stop = True
						print(i,j,k,nnX,nnY,nnZ,lattice[i][j][k].tensor().shape())
			for i in range(nnX):
				lattice[i] = lattice[i][::2]
			nnY = len(lattice[0])
			for i in range(nnX):
				for j in range(nnY):
					for k in range(nnZ):
							makeTop(lattice[i][j][k])
		print(network)
		network.compressLinks()
		print(network)

		if nnX == 1 and nnY == 1 and nnZ == 1 or stop:
			break

		print(len(network.topLevelNodes()))

		if nnZ > 1:
			for kk in range(nnZ//2):
				for i in range(nnX):
					for j in range(nnY):
						k = kk*2+(counter%2)
						for q in range(3):
							for p in range(3):
								for t in range(3):
									lattice[(i+q-1)%nnX][(j+p-1)%nnY][(k+t-1)%nnZ] = makeTop(lattice[(i+q-1)%nnX][(j+p-1)%nnY][(k+t-1)%nnZ])
						if lattice[i][j][k] in lattice[i][j][(k+1)%nnZ].connectedHigh():
							lattice[i][j][k].merge(lattice[i][j][(k+1)%nnZ],compressL=True,mergeL=True,eps=1e-4)
						else:
							stop = True
						print(i,j,k,nnX,nnY,nnZ,lattice[i][j][k].tensor().shape())
			for i in range(nnX):
				for j in range(nnY):
					lattice[i][j] = lattice[i][j][::2]
			nnZ = len(lattice[0][0])
			for i in range(nnX):
				for j in range(nnY):
					for k in range(nnZ):
							makeTop(lattice[i][j][k])
		print(network)

		counter += 1

	network.contract(compressL=True,mergeL=True,eps=1e-4)

	for i in list(network.topLevelNodes()):
		print(i.tensor().array())

	return np.log(list(network.topLevelNodes())[0].tensor().array()) + list(network.topLevelNodes())[0].logScalar()

print(otherIsingSolve(5,5,5,0.2,0.2,-5.0))
#cProfile.run('otherIsingSolve(4,4,4,-0.2,-0.2,1.)')
exit()

