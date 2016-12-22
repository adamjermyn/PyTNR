import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds
from itertools import combinations

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

# A note on precision:
# the precision option here allows you to truncate the svd when the desired accuracy
# is achieved. This is done by comparing the sum of the valued obtained so far to the
# frobenius norm ofthe matrix.

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

def rank4InverseSVD(arr1, arr2):
	'''
	This method takes two rank-3 arrays arr1 and arr2.
	Let their indices be ijk and lmk respectively.
	These arrays must have matching size along the last index,
	such that arr2.shape[-1] == arr2.shape[-1].

	Given such arrays, this matrix computes a pair of arrays
	arr3 and arr4 with indices ilk and jmk, such that once more
	they match in size along the last axis. These arrays
	are chosen so as to give the same inner product as arr1 and arr2,
	just reshuffled (i.e. arr1 and arr2 give ijlm while arr3 and arr4
	give iljm).

	The reason this is useful is that loop elimination often involves
	'flipping' which bonds are together on the same tensor in a pair.
	'''
	shape = (arr1.shape[0]*arr2.shape[0],arr1.shape[1]*arr2.shape[1])

	# There are three possible contraction orders,
	# and we need to figure out which is most efficient.
	s1 = arr1.shape[1]*arr1.shape[2]*arr2.shape[0] + arr2.shape[0]*arr2.shape[1]*arr2.shape[2]
	s2 = arr1.shape[0]*arr1.shape[1]*arr1.shape[2] + arr1.shape[0]*arr2.shape[1]*arr2.shape[2]
	s3 = arr1.shape[0]*arr1.shape[1]*arr2.shape[0]*arr2.shape[1] + arr1.shape[0]*arr2.shape[0]

	if s1 < s2 and s1 < s3:
		def matvec(v):
			v = np.reshape(v, (arr1.shape, arr2.shape))
			bcd = np.einsum('ijk,il->jkl',arr1,v)
			vr = np.einsum('jkl,lmk->jm',bcd,arr2)
			return vr
	elif s2 < s1 and s2 < s3:
		def matvec(v):
			v = np.reshape(v, (arr1.shape, arr2.shape))
			abde = np.einsum('ijk,lmk->ijlm',arr1,arr2)
			vr = np.einsum('ijlm,il->jm',abde,v)
			return vr
	else:
		def matvec(v):
			v = np.reshape(v, (arr1.shape, arr2.shape))
			ace = np.einsum('il,lmk->imk',v,arr2)
			vr = np.einsum('imk,ijk->jm',ace,arr1)
			return vr

	linop = LinearOperator(shape, matvec=matvec)

	u, lam, v, p, cp = generalSVD(linop)

	

def entropy(array, pref=None, tol=1e-3):
	'''
	This method determines the best pair of indices to split off.
	That pair is just the one with the minimum entropy to the rest of the indices.

	This is determined by iteratively refining bounds of the entropy associated with
	each possible pair and throwing away provably worse options until only one remains.

	To avoid refining endlessly when options are essentially identical this method
	takes as optional input a tolerance tol below which it does not care about
	entropy differences. This defaults to 1e-3 and is treated as an absolute tolerance.
	That is, the returned answer is guaranteed to be optimal within this tolerance.

	This method also takes as an optional input pref, which specifies a tie-breaking
	preference in cases where multiple options lie within tol of each other and the
	optimum. This should be specified as a set containing a pair of indices.
	'''

	# Make sure pref is a set:
	if pref is None:
		pref = set()
	else:
		pref = set(pref)


	# Generate list of pairs of indices
	indexLists = list(combinations(range(len(array.shape)), 2))

	# We filter out options which are complements of one another, and
	# hence give the same answer. We do not filter out pref in this process.

	indexLists = [set(q) for q in indexLists]

	complements = [set(range(len(array.shape))).difference(l) for l in indexLists]
	indexSets = [set(l) for l in indexLists]
	while len(complements) > 0:
		c = complements.pop()
		if c in indexSets and c != pref:
			indexSets.remove(c)
			s = set(range(len(array.shape)))
			s = s.difference(c)
			if s in complements:
				complements.remove(s)
	indexLists = [tuple(l) for l in indexSets]

	# Lists for storing intermediate results.
	mins = [1e10 for _ in indexLists]			# Lower bound on entropy
	maxs = [-1 for _ in indexLists]				# Upper bound on entropy
	norms = [-1 for _ in indexLists]			# Frobenius norms of the array in different shapes
	knownVals = [None for _ in indexLists]		# Temporary storage for singular values
	liveIndices = list(range(len(indexLists)))	# Stores the indices of options which have not been ruled out.

	bondDimension = 1		# We start with just two singular values (set to 1 so that it becomes 2 upon doubling)
	lowest = [1e10,-1]		# Keeps track of the index with the lowest upper bound on the entropy
	while len(liveIndices) > 1:
		bondDimension *= 2	# Double the bond dimension

		for i in list(liveIndices): # We copy the list so we can remove from it while looping
			indices = indexLists[i]

			# Put the array in the right shape
			arr = permuteIndices(array, indices)
			sh = arr.shape[:len(indices)]
			s = np.product(sh)
			arr = np.reshape(arr, (s,-1))

			# Calculate the norm if it hasn't been done already
			if norms[i] == -1:
				norms[i] = np.linalg.norm(arr)

			# If the bond dimension is too large, full SVD is required.
			if bondDimension > min(arr.shape) - 1:
				lams = np.linalg.svd(arr, full_matrices=0, compute_uv=False)
			else:
				lams = bigSVDvals(arr, bondDimension)

			lams /= norms[i]					# Normalize
			knownVals[i] = lams**2				# Turn into probabilities
			p = knownVals[i]
			p = p[p>0]							# Ensure probabilities are non-zero for floating point reasons
			mins[i] = -np.sum(p*np.log(p))		# Compute entropy of probabilities
			maxs[i] = mins[i]

			# If there is left-over probability we get additional entropy,
			# but we don't know how much so we just calculate bounds.
			q = 1 - np.sum(p)
			if q > 0 and bondDimension < min(arr.shape):
				mins[i] -= q*np.log(q)		# Corresponds to a single singular value holding the remaining probability
				maxs[i] -= q*np.log(q/(min(arr.shape)-bondDimension)) # Corresponds to multiple singular values holding it

			# Now we check if any can be eliminated
			if maxs[i] < lowest[0] - tol:		# Means this is better than the previous best
				lowest[0] = maxs[i]
				lowest[1] = i
			elif mins[i] > lowest[0] + tol:		# Means that this is strictly worse than the current best
				liveIndices.remove(i)
			else:								# Means this is tied within tolerance to the current best
				if pref == set():				# If we have no preference we remove this unless it is the current best
					if i != lowest[1]:
						liveIndices.remove(i)
				elif pref != set(indexLists[i]) and pref in [indexSets[j] for j in liveIndices]:
												# Otherwise we remove this so long as the preferred option is still live
												# and this is not it.
					liveIndices.remove(i)
	return list(indexLists[liveIndices[0]])

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

