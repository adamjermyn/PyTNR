from tensor import Tensor
from tensor import matrixToTensor, tensorToMatrix
from utils import matrixProductLinearOperator, generalSVD
from utils import tupleReplace
import numpy as np

def cutBond(u, v, ind1, ind2, link):
	n1, n2, _, _, sh1, sh2 = link.topContents()

	sh1m = tupleReplace(sh1, ind1, None)
	sh2m = tupleReplace(sh2, ind2, None)

	u = np.reshape(u, sh1m)
	v = np.reshape(v, sh2m)

	t1m = Tensor(u.shape, u)
	t2m = Tensor(v.shape, v)

	n1m = n1.modify(t1m, delBuckets=[ind1])
	n2m = n2.modify(t2m, delBuckets=[ind2])

	link.network().registerLinkCut(link)
	link.setParent(link) # So it is unambiguously not top-level.
	link.setCompressed()

	return link, n1m, n2m

def compress(link, optimizerArray=None, optimizerBuckets=None, eps=1e-2):
	n1, n2, t1, t2, sh1, sh2 = link.topContents()

	arr1 = t1.array()
	arr2 = t2.array()

	ind1 = n1.bucketIndex(link.bucket1())
	ind2 = n2.bucketIndex(link.bucket2())

	shI = sh1[ind1] # Must be the same as arr2.shape[ind2]

	if shI == 1: # Means we just cut the bond 
		return cutBond(arr1, arr2, ind1, ind2, link) 

	arr11 = tensorToMatrix(t1, ind1, front=False)
	arr22 = tensorToMatrix(t2, ind2, front=True)

	opN = matrixProductLinearOperator(arr11, arr22)

	if optimizerArray is None:
		optimizerMatrix = None
	else:
		# First we need to establish correspondence between indices in
		# optimizerArray and arr1.

		inds1 = {}	# Stores the indices in arr1 as values
		inds2 = {}	# Stores the indices in arr2 as values
		inds  = {}	# Stores the indices in optimizerArray as values

		print(len(optimizerBuckets))

		for i,b in enumerate(optimizerBuckets):
			if b in n1.buckets():
				j = n1.bucketIndex(b)
				inds1[i] = j
				inds[(1,j)] = i
				print(1,j,i)
		for i,b in enumerate(optimizerBuckets):
			if b in n2.buckets():
				j = n2.bucketIndex(b)
				inds2[i] = j
				inds[(2,j)] = i
				print(2,j,i)

		print(len(inds1),len(inds2),len(n1.buckets()),len(n2.buckets()))
		print(len(inds))
		print(inds)
		print(ind1,ind2)
		# Now we put all indices corresponding to arr1 at the front,
		# and all indices corresponding to arr2 at the back.
		perm = []

		for i in range(len(arr1.shape)):
			if i != ind1:
				perm.append(inds[(1,i)])
		for i in range(len(arr2.shape)):
			if i != ind2:
				perm.append(inds[(2,i)])

		optimizerMatrix = np.transpose(optimizerArray, axes=perm)

		sh1m = np.product(tupleReplace(sh1, ind1, None))
		sh2m = np.product(tupleReplace(sh2, ind2, None))

		optimizerMatrix = np.reshape(optimizerMatrix, (sh1m, sh2m))

	u, lam, v, _, cp = generalSVD(opN, bondDimension=min(sh1[ind1], min(opN.shape)-1), optimizerMatrix=optimizerMatrix)

	ind = np.searchsorted(cp, eps, side='left')
	ind = len(cp) - ind

	if ind == len(cp): 
		# This means that we can't compress this bond, and so
		# we leave it untouched to avoid incurring floating point error.
		link.setCompressed()
		if optimizerArray is not None:
			link.setOptimized()
		return link, n1, n2
	else:
		# Means that we will either compress or cut the Link.
		u = np.transpose(u)

		lam = lam[:ind]
		u = u[:ind, :]
		v = v[:ind, :]

		u *= np.sqrt(lam[:, np.newaxis])
		v *= np.sqrt(lam[:, np.newaxis])

		if ind > 1:
			t1m = matrixToTensor(u, tupleReplace(arr1.shape, ind1, ind), ind1, front=True)
			t2m = matrixToTensor(v, tupleReplace(arr2.shape, ind2, ind), ind2, front=True)

			n1m = n1.modify(t1m, repBuckets=[ind1])
			n2m = n2.modify(t2m, repBuckets=[ind2])

			optimized = (optimizerArray is not None)

			newLink = n1m.addLink(n2m, ind1, ind2, compressed=True, optimized=optimized, children=[link])
			return newLink, n1m, n2m
		else:	# Means we're just cutting the bond
			return cutBond(u, v, ind1, ind2, link)

