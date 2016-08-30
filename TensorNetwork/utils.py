import numpy as np
from tensor import Tensor
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds

################################
# Miscillaneous Helper Functions
################################

def tupleReplace(tpl, i, j):
	'''
	Returns a tuple with element i of tpl replaced with the quantity j.
	'''
	tpl = list(tpl)
	tpl = tpl[:i] + [j] + tpl[i+1:]
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

###################################
# Linear Operator and SVD Functions
###################################

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

def generalSVD(matrix, bondDimension=np.inf):
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
	'''

	if bondDimension > 0 and bondDimension < matrix.shape[0] and bondDimension < matrix.shape[1]:
		# Required so sparse bond is properly represented
		return bigSVD(matrix, bondDimension)
	else:
		return np.linalg.svd(matrix, full_matrices=0)

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
		arr = tens.array()
		shape = tens.shape()

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

def matrixToTensor(matrix, shape, index, front=True):
		'''
		This method takes a 2D array and reshapes it to the given shape.
		The reshape operation only modifies one of the axes of the matrix.
		This is either the first (front) or last (not front) depending on the
		boolean variable front. Whichever index is not reshaped is then
		put in the position specified by index.

		This method is meant to be the inverse of tensorToMatrix.
		'''
		if not front:
			matrix = np.transpose(matrix)

		shm = shape[:index] + shape[index+1:]

		matrix = np.reshape(matrix, [shape[index]] + list(shm))

		perm = list(range(len(shape)))
		perm = perm[1:]
		perm.insert(index, 0)

		matrix = np.transpose(matrix, axes=perm)

		t = Tensor(matrix.shape, matrix)

		return t


