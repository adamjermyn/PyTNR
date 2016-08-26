from tensor import Tensor
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator

def makeLinearOperator(matrix1, matrix2):
	'''
	The reason we implement our own function here is that the dot product
	associated with the standard LinearOperator class has an extremely slow
	type-checking stage which has to be performed every time a product is calculated.
	'''

	shape = (matrix1.shape[0],matrix2.shape[1])

	def matvec(v):
		return np.dot(matrix1,np.dot(matrix2,v))

	def matmat(m):
		return np.dot(matrix1,np.dot(matrix2,m))

	def rmatvec(v):
		return np.dot(np.transpose(matrix2),np.dot(np.transpose(matrix1),v))

	return LinearOperator(shape, matvec=matvec, matmat=matmat, rmatvec=rmatvec)



def cutBond(u, v, n1, n2, ind1, ind2, link, sh1m, sh2m):
	u = np.reshape(u,sh1m)
	v = np.reshape(v,sh2m)

	t1m = Tensor(u.shape,u)
	t2m = Tensor(v.shape,v)

	n1m = n1.modify(t1m, delBuckets=[ind1])
	n2m = n2.modify(t2m, delBuckets=[ind2])

	link.network().registerLinkCut(link)
	link.setParent(link) # So it is unambiguously not top-level.
	link.setCompressed()
	link.update() # So it's up to date.

	return link, n1m, n2m

def bigSVD(matrix, k):
	u, s, v = svds(matrix, k=k)
	inds = np.argsort(s)
	inds = inds[::-1]
	u = u[:,inds]
	s = s[inds]
	v = v[inds,:]
	return u, s, v

def compress(link, eps=1e-4):
	n1 = link.bucket1().topNode()
	n2 = link.bucket2().topNode()

	t1 = n1.tensor()
	t2 = n2.tensor()

	arr1 = t1.array()
	arr2 = t2.array()

	sh1 = list(arr1.shape)
	sh2 = list(arr2.shape)

	ind1 = n1.bucketIndex(link.bucket1())
	ind2 = n2.bucketIndex(link.bucket2())

	sh1m = sh1[:ind1] + sh1[ind1+1:]
	sh2m = sh2[:ind2] + sh2[ind2+1:]

	shI = arr1.shape[ind1] # Must be the same as arr2.shape[ind2]

	if shI == 1: # Means we just cut the bond
		return cutBond(np.copy(arr1), np.copy(arr2), n1, n2, ind1, ind2, link, sh1m, sh2m)

	if np.product(sh1m) > shI and np.product(sh2m) > shI: # Required so sparse bond is properly represented
		perm = range(len(arr1.shape))
		perm.remove(ind1)
		perm.append(ind1)
		arr11 = np.transpose(arr1, axes=perm)
		arr11 = np.reshape(arr11,[np.product(sh1m),shI])

		perm = range(len(arr2.shape))
		perm.remove(ind2)
		perm.insert(0, ind2)
		arr22 = np.transpose(arr2, axes=perm)
		arr22 = np.reshape(arr22,[shI,np.product(sh2m)])

		opN = makeLinearOperator(arr11, arr22)

		u, lam, v = bigSVD(opN, min(sh1[ind1], min(opN.shape)-1))
	else:
		cont = t1.contract(ind1,t2,ind2)
		arrN = cont.array()
		arrN = np.reshape(arrN,(np.product(sh1m),np.product(sh2m)))
		u, lam, v = np.linalg.svd(arrN,full_matrices=0)

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	ind = np.searchsorted(cp, eps, side='left')
	ind = len(cp) - ind

	if ind == len(cp): 	# Means we won't actually compress it
		link.setCompressed()
		return link, n1, n2
	else:				# Means we will compress it
		u = np.transpose(u)

		lam = lam[:ind]
		u = u[:ind,:]
		v = v[:ind,:]

		u *= np.sqrt(lam[:,np.newaxis])
		v *= np.sqrt(lam[:,np.newaxis])

		if ind > 1:
			u = np.reshape(u,[ind] + sh1m)
			v = np.reshape(v,[ind] + sh2m)

			perm1 = range(len(arr1.shape))
			perm1 = perm1[1:]
			perm1.insert(ind1,0)
			perm2 = range(len(arr2.shape))
			perm2 = perm2[1:]
			perm2.insert(ind2,0)

			u = np.transpose(u,axes=perm1)
			v = np.transpose(v,axes=perm2)

			t1m = Tensor(u.shape,u)
			t2m = Tensor(v.shape,v)

			n1m = n1.modify(t1m, repBuckets=[ind1])
			n2m = n2.modify(t2m, repBuckets=[ind2])

			newLink = n1m.addLink(n2m, ind1, ind2, compressed=True, children=[link])

		else:	# Means we're just cutting the bond
			return cutBond(u, v, n1, n2, ind1, ind2, link, sh1m, sh2m)

	return newLink, n1m, n2m