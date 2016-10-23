import numpy as np
from numpy.linalg import svd
from scipy.linalg import logm
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds

################################
# Miscillaneous Helper Functions
################################

def tupleReplace(tpl, i, j):
	'''
	Returns a tuple with element i of tpl replaced with the quantity j.
	If j is None, just removes element i.
	'''
	assert i >= 0
	assert i < len(tpl)

	tpl = list(tpl)
	if j is not None:
		tpl = tpl[:i] + [j] + tpl[i+1:]
	else:
		tpl = tpl[:i] + tpl[i+1:]
	return tuple(tpl)


##################################
# General Linear Algebra Functions
##################################

def kroneckerDelta(dim, length):
	'''
	Returns the Kronecker delta of dimension dim and length length.
	Note that if dim == 0 the scalar 1. is returned. If dim == 1 a
	vector of ones is returned. These are the closest conceptual
	analogues of the Kronecker delta, and work in a variety of circumstances
	where a Kronecker delta is the natural higher-dimensional generalisation.
	'''
	if dim == 0:
		return 1
	elif dim == 1:
		return np.ones(length)
	elif dim > 1:
		arr = np.zeros(tuple(length for i in range(dim)))
		np.fill_diagonal(arr,1.0)
		return arr

def adjoint(m):
	return np.transpose(np.conjugate(m))

#######################
# Tensor Helper Methods
#######################

def tensorToMatrix(tens, index, front=True):
		'''
		This method flattens the Tensor's array along all indices other than
		index and does so in a way which preserves the ordering of the other
		axes when unflattened.

		This method also takes as input a boolean variable front. If front is True
		then the special index is pushed to the beginning. If front is False then the
		special index is pushed to the back.
		'''
		return ndArrayToMatrix(tens.array(), index, front=front)

def matrixToTensor(matrix, shape, index, front=True):
		'''
		This method takes a 2D array and reshapes it to the given shape.
		The reshape operation only modifies one of the axes of the matrix.
		This is either the first (front) or last (not front) depending on the
		boolean variable front. Whichever index is not reshaped is then
		put in the position specified by index.

		This method is meant to be the inverse of tensorToMatrix.
		'''
		arr = matrixToNDArray(matrix, shape, index, front=front)
		return Tensor(arr.shape, arr)

###################################
# Linear Operator and SVD Functions
###################################

def permuteIndices(arr, indices):
	'''
	This method moves the indices specified in indices
	to be the first ones in the array.
	'''
	shape = arr.shape
	perm = list(range(len(shape)))
	for i in indices:
		perm.remove(i)
	for j,i in enumerate(indices):
		perm.insert(j, i)
	return np.transpose(arr, axes=perm)

def ndArrayToMatrix(arr, index, front=True):
		'''
		This method flattens the array along all indices other than
		index and does so in a way which preserves the ordering of the other
		axes when unflattened.

		This method also takes as input a boolean variable front. If front is True
		then the special index is pushed to the beginning. If front is False then the
		special index is pushed to the back.
		'''
		shape = arr.shape

		perm = list(range(len(shape)))
		perm.remove(index)

		shm = shape[:index] + shape[index+1:]
		shI = shape[index]

		if front:
			perm.insert(0, index)
			arr = np.transpose(arr, axes=perm)
			arr = np.reshape(arr, [shI, np.product(shm)])
		else:
			perm.append(index)
			arr = np.transpose(arr, axes=perm)
			arr = np.reshape(arr, [np.product(shm), shI])

		return arr

def matrixToNDArray(matrix, shape, index, front=True):
		'''
		This method takes a 2D array and reshapes it to the given shape.
		The reshape operation only modifies one of the axes of the matrix.
		This is either the first (front) or last (not front) depending on the
		boolean variable front. Whichever index is not reshaped is then
		put in the position specified by index.

		This method is meant to be the inverse of ndArrayToMatrix.
		'''
		if not front:
			matrix = np.transpose(matrix)

		shm = shape[:index] + shape[index+1:]

		matrix = np.reshape(matrix, [shape[index]] + list(shm))

		perm = list(range(len(shape)))
		perm = perm[1:]
		perm.insert(index, 0)

		matrix = np.transpose(matrix, axes=perm)

		return matrix

def matrixProductLinearOperator(matrix1, matrix2):
	'''
	The reason we implement our own function here is that the dot product
	associated with the standard LinearOperator class has an extremely slow
	type-checking stage which has to be performed every time a product is calculated.
	'''

	if matrix1.shape[0] < matrix1.shape[1] or matrix2.shape[1] < matrix2.shape[0]:
		return np.dot(matrix1, matrix2)

	shape = (matrix1.shape[0],matrix2.shape[1])

	def matvec(v):
		return np.dot(matrix1,np.dot(matrix2,v))

	def matmat(m):
		return np.dot(matrix1,np.dot(matrix2,m))

	def rmatvec(v):
		return np.dot(np.transpose(matrix2),np.dot(np.transpose(matrix1),v))

	return LinearOperator(shape, matvec=matvec, matmat=matmat, rmatvec=rmatvec)

def bigSVD(matrix, bondDimension):
	'''
	This is a helper method for wrapping the iterative SVD method scipy provides.
	It takes as input a matrix and an integer bond dimension which corresponds
	to the bond which was contracted to form this matrix. It returns the SVD
	with the singular values sorted in descending order (which the iterative
	solver does not on its own guarantee).
	'''
	u, s, v = svds(matrix, k=bondDimension)
	inds = np.argsort(s)
	inds = inds[::-1]
	u = u[:, inds]
	s = s[inds]
	v = v[inds, :]
	return u, s, v

def generalSVD(matrix, bondDimension=np.inf, optimizerMatrix=None, arr1=None, arr2=None):
	'''
	This is a helper method for SVD calculations.

	In the case where the matrix of interest was formed as a product A*B where
	A.shape[1] == B.shape[0] << A.shape[0], B.shape[1], the SVD is mathematically
	guaranteed to return at most A.shape[1] nonzero singular values, and so
	we ought to use an iterative solver rather than the full solver.

	In the remaining cases the iterative solve will fail because it cannot retrieve
	all singular values of a matrix (it can retrieve at most one fewer). This is
	just an implementation detail, and only arises in rare cases, so we just revert
	to the full SVD solve.

	If an optimizerArray is provided, we perform the procedure outlined in Eq7-13
	in "Second Renormalization of Tensor-Network States" by Xie et. al. (arXiv:0809.0182v3).

	This method accepts as input:
		matrix 			-	The matrix to SVD.
		bondDimension	-	The bond dimension used to form the matrix (i.e. A.shape[1]).
		optimizerTensor	-	The 'environment' array to optimize against.

	This method returns:
		u 	-	A matrix of the left singular vectors
		lam	-	An array of the singular values
		v 	-	An array of the right singular vectors
		p 	-	An array whose entries reflect the portion of the total weight in each
				singular value.
		cp 	-	An array whose entries reflect the cumulative portion of the total weight
				associated with all singular values past a given index.

	As a result of the above definitions, p and cp are both sorted in descending order.
	'''
	if arr1 is not None and arr2 is not None:
		u1, s1, v1 = np.linalg.svd(arr1, full_matrices=0)
		print('SVD 1 Done!')
		u2, s2, v2 = np.linalg.svd(arr2, full_matrices=0)
		print('SVD 2 Done!')

		arr3 = np.dot(v1, u2)
		arr3 = np.einsum('i,ij,j->ij',s1,arr3,s2)

		up, lam, vp = np.linalg.svd(arr3, full_matrices=0)
		print('SVD 3 Done!')
		u = np.dot(u1, up)
		v = np.dot(vp, v2)
	elif optimizerMatrix is None:
		if bondDimension > 0 and bondDimension < matrix.shape[0] and bondDimension < matrix.shape[1]:
			# Required so sparse bond is properly represented
			u, lam, v = bigSVD(matrix, bondDimension)
		else:
			u, lam, v = np.linalg.svd(matrix, full_matrices=0)
	else:
		ue, lame, ve = np.linalg.svd(optimizerMatrix, full_matrices=0)

		lams = np.sqrt(lame)

		print(lams.shape, ve.shape, matrix.shape, ue.shape, lams.shape)

		mmod = lams*np.dot(ve,np.dot(matrix,ue))*lams

		um, lamm, vm = np.linalg.svd(mmod, full_matrices=0)

		lamms = np.sqrt(lamm)

		uf = np.einsum('ij,j,jk->ik',adjoint(ve),1/lamms,um)
		vf = np.einsum('ij,j,jk->ik',vm,1/lamms,adjoint(ue))

		u, lam, v = uf, lamm, vf

		assert u.shape[0] == matrix.shape[0]
		assert v.shape[1] == matrix.shape[1]

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	return u, lam, v, p, cp

def entropy(array, indices):
	arr = permuteIndices(array, indices)
	sh = arr.shape[:len(indices)]
	s = np.product(sh)
	arr = np.reshape(arr, (s,-1))
	u, lam, v, p, cp = generalSVD(arr)

	s = -np.sum(p*np.log(p))

	return s

def splitArray(array, chunkIndices, accuracy=1e-4):
	perm = []

	c1 = 0
	c2 = 0
	sh1 = []
	sh2 = []
	indices1 = []
	indices2 = []

	for i in range(len(array.shape)):
		if i in chunkIndices:
			perm.append(c1)
			sh1.append(array.shape[i])
			indices1.append(i)
			c1 += 1
		else:
			perm.append(len(chunkIndices) + c2)
			sh2.append(array.shape[i])
			indices2.append(i)
			c2 += 1

	array2 = np.transpose(array, axes=perm)

	array2 = np.reshape(array2, (np.product(sh1),np.product(sh2)))

	u, lam, v = svd(array2, full_matrices=0)

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	ind = np.searchsorted(cp, accuracy, side='left')
	ind = len(cp) - ind

	u = u[:,:ind]
	lam = lam[:ind]
	v = v[:ind,:]

	u *= np.sqrt(lam)[np.newaxis,:]
	v *= np.sqrt(lam)[:,np.newaxis]

	u = np.reshape(u, sh1 + [ind])
	v = np.reshape(v, [ind] + sh2)

	assert u.shape[-1] == v.shape[0]

	return u,v,indices1,indices2



