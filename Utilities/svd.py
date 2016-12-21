import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds

from TNRG.Utilities.arrays import permuteIndices
from TNRG.Utilities.linalg import adjoint

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

def bigSVDvals(matrix, bondDimension):
	'''
	This is a helper method for wrapping the iterative SVD method scipy provides.
	It takes as input a matrix and an integer bond dimension which corresponds
	to the bond which was contracted to form this matrix. It returns the SVD
	with the singular values sorted in descending order (which the iterative
	solver does not on its own guarantee).
	'''
	s = svds(matrix, k=bondDimension, return_singular_vectors=False)
	inds = np.argsort(s)
	inds = inds[::-1]
	s = s[inds]
	return s




def generalSVD(matrix, bondDimension=np.inf, optimizerMatrix=None, arr1=None, arr2=None, precision=1e-5):
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
		u2, s2, v2 = np.linalg.svd(arr2, full_matrices=0)

		arr3 = np.dot(v1, u2)
		arr3 = np.einsum('i,ij,j->ij',s1,arr3,s2)

		up, lam, vp = np.linalg.svd(arr3, full_matrices=0)
		u = np.dot(u1, up)
		v = np.dot(vp, v2)
	elif precision is not None and bondDimension is np.inf and min(matrix.shape) > 4:
		# Matches Frobenius norm
		norm = np.linalg.norm(matrix)**2
		err = norm

		bondDimension = 2

		while err > precision:
			print('Matrix:',matrix.shape)
			print('BD:',bondDimension)
			bondDimension *= 2
			if bondDimension > min(matrix.shape) - 1:
				u, lam, v = np.linalg.svd(matrix, full_matrices=0)
				err = 0
			else:
				u, lam, v = bigSVD(matrix, bondDimension)
				err = abs(norm - np.sum(lam**2))
		if err == 0:
			print('BD:',min(matrix.shape),min(matrix.shape))
		else:
			print('BD:',bondDimension,min(matrix.shape))
	elif optimizerMatrix is None:
		if bondDimension > 0 and bondDimension < matrix.shape[0] and bondDimension < matrix.shape[1]:
			# Required so sparse bond is properly represented
			u, lam, v = bigSVD(matrix, bondDimension)
		else:
			try:
				u, lam, v = np.linalg.svd(matrix, full_matrices=0)
			except:
				print('Erm.... SVD not converged!')
				print(matrix)
				print(np.sum(matrix**2))
				print(matrix.shape)
				print(np.max(matrix), np.min(matrix))
				np.savetxt('error',matrix)
				exit()
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

def generalSVDvals(matrix, bondDimension=np.inf, optimizerMatrix=None, precision=1e-5):
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
		lam	-	An array of the singular values
		p 	-	An array whose entries reflect the portion of the total weight in each
				singular value.
		cp 	-	An array whose entries reflect the cumulative portion of the total weight
				associated with all singular values past a given index.

	As a result of the above definitions, p and cp are both sorted in descending order.
	'''
	if precision is not None and bondDimension is np.inf and min(matrix.shape) > 4:
		# Matches Frobenius norm
		norm = np.linalg.norm(matrix)**2
		err = norm

		bondDimension = 2

		while err > precision:
			print('Matrix:',matrix.shape)
			print('BD:',bondDimension)
			bondDimension *= 2
			if bondDimension > min(matrix.shape) - 1:
				lam = np.linalg.svd(matrix, full_matrices=0, compute_uv=False)
				err = 0
			else:
				lam = bigSVDvals(matrix, bondDimension)
				err = abs(norm - np.sum(lam**2))
		if err == 0:
			print('BD:',min(matrix.shape),min(matrix.shape))
		else:
			print('BD:',bondDimension,min(matrix.shape))
	elif optimizerMatrix is None:
		if bondDimension > 0 and bondDimension < matrix.shape[0] and bondDimension < matrix.shape[1]:
			# Required so sparse bond is properly represented
			lam = bigSVDvals(matrix, bondDimension)
		else:
			try:
				lam = np.linalg.svd(matrix, full_matrices=0, compute_uv=False)
			except:
				print('Erm.... SVD not converged!')
				print(matrix)
				print(np.sum(matrix**2))
				print(matrix.shape)
				print(np.max(matrix), np.min(matrix))
				np.savetxt('error',matrix)
				exit()

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	return lam, p, cp

def entropy(array, indices):
	arr = permuteIndices(array, indices)
	sh = arr.shape[:len(indices)]
	s = np.product(sh)
	arr = np.reshape(arr, (s,-1))
	lam, p, cp = generalSVDvals(arr, precision=1e-5)
	s = -np.sum(p*np.log(p))
	return s

def splitArray(array, indices, accuracy=1e-4):
	perm = []

	sh1 = [array.shape[i] for i in indices]
	sh2 = [array.shape[i] for i in range(len(array.shape)) if i not in indices]
	indices1 = list(indices)
	indices2 = [i for i in range(len(array.shape)) if i not in indices]

	arr = permuteIndices(array, indices)
	arr = np.reshape(arr, (np.product(sh1), np.product(sh2)))
	u, lam, v, p, cp = generalSVD(arr)

	ind = np.searchsorted(cp, accuracy, side='left')
	ind = len(cp) - ind

	u = u[:,:ind]
	lam = lam[:ind]
	v = v[:ind,:]

	u *= np.sqrt(lam)[np.newaxis,:]
	v *= np.sqrt(lam)[:,np.newaxis]

	u = np.reshape(u, sh1 + [ind])
	v = np.reshape(v, [ind] + sh2)

	return u,v,indices1,indices2

