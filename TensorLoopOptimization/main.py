import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, lsqr

def shift(l, n):
	'''
	Shifts the list l forward by n indices.
	'''

	n = n % len(l)
	return l[-n:] + l[:-n]

def contract(t1, t2):
	'''
	t1 and t2 are lists of rank-3 tensors set such that in each list the last index of
	each contracts with the first index of the next, and the last index of the last tensor
	contracts with the first index of the first one.

	The return value is the inner product of these tensors.
	'''

	# Contract the connections between the two lists
	cont = [np.tensordot(a,b, axes=((1,),(1,))) for a,b in zip(*(t1, t2))]

	# Contract the two lists
	x = cont[0]
	x = np.swapaxes(x, 1, 2)
	for y in cont[1:]:
		x = np.tensordot(x, y, axes=((2,3),(0,2)))

	# Contract the periodic indices
	n = np.einsum('ijij->',x)

	return n

def norm(tensors):
	'''
	tensors is a list of rank-3 tensors set such that the last index of each contracts
	with the first index of the next, and the last index of the last tensor contracts
	with the first index of the first one.

	The return value is the L2 norm of the tensor.
	'''
	return contract(tensors, tensors)

def normOp(tensors, index):
	'''
	tensors is a list of rank-3 tensors set such that the last index of each contracts
	with the first index of the next, and the last index of the last tensor contracts
	with the first index of the first one.

	The return value is a sparse linear operator which represents the action of the tensor
	minus that at index contracted against itself minus that at index. This is the operator
	N from equation S10 in arXiv:1512.04938.
	'''
	
	# Contract the connections between the two lists
	cont = [np.tensordot(a,b, axes=((1,),(1,))) for a,b in zip(*(tensors, tensors))]
	
	# Rotate cont so that index becomes len(cont)-1.
	cont = shift(cont, len(cont) - 1 - index)

	# Contract the two lists aside from index
	x = cont[0]
	x = np.swapaxes(x, 1, 2)
	for y in cont[1:len(cont)-1]:
		x = np.tensordot(x, y, axes=((2,3),(0,2)))

	# Construct N as a sparse linear operator.
	def matvec(v):
		# v is a flattened version of a rank-3 tensor.
		v = np.reshape(v, tensors[index].shape)
		ret = np.tensordot(x, v, axes=((2,0),(0,2)))
		ret = np.swapaxes(ret, 1, 2)

		# At this point a careful counting suggests that we should swap indices 0 and 2.
		# A more careful counting, however, reveals that because these swap upon moving from
		# the space to its dual, we don't need to do so.
		return ret

	def rmatvec(v):
		# This is a square operator.
		return matvec(v)

	op = LinearOperator((tensors[index].size,tensors[index].size), matvec=matvec, rmatvec=rmatvec)

	return op

def normMat(tensors, index):
	'''
	tensors is a list of rank-3 tensors set such that the last index of each contracts
	with the first index of the next, and the last index of the last tensor contracts
	with the first index of the first one.

	The return value is a matrix which represents the action of the tensor
	minus that at index contracted against itself minus that at index. This is the operator
	N from equation S10 in arXiv:1512.04938.
	'''
	# Contract the connections between the two lists
	cont = [np.tensordot(a,b, axes=((1,),(1,))) for a,b in zip(*(tensors, tensors))]
	
	# Rotate cont so that index becomes len(cont)-1.
	cont = shift(cont, len(cont) - 1 - index)

	# Contract the two lists aside from index
	x = cont[0]
	x = np.swapaxes(x, 1, 2)
	for y in cont[1:len(cont)-1]:
		x = np.tensordot(x, y, axes=((2,3),(0,2)))

	iden = np.identity(tensors[index].shape[1])
	x = np.tensordot(x, iden, axes=(tuple(),tuple()))
	x = np.transpose(x, axes=(3,5,0,2,4,1))
	x = np.reshape(x, (tensors[index].size, tensors[index].size))

	return x

def optimizeTensor(t1, t2, index, eps=1e-5):
	'''
	t1 and t2 are lists of rank-3 tensors set such that in each list the last index of
	each contracts with the first index of the next, and the last index of the last tensor
	contracts with the first index of the first one.

	The return value is a version of t2 which maximizes the inner product of t1 and t2, subject
	to the constraint that the norm of t1 equals that of t2. In the optimization process only
	the tensor at the specified index may be modified.

	Following equation S10 in arXiv:1512.04938, we first compute two tensors: N and W.

	N is given by contracting all of t2 against itself except for the tensors at the specified index.
	This yields a rank-6 object, where two indices arise from the implicit identity present in the bond
	between the two copies of t2[index].
	
	W is given by contracting all of t2 (other than t2[index]) against t1. This yields a rank-3 object.

	We then let
	
	N . t2[index] = W

	and solve for t2[index]. This is readily phrased as a matrix problem by flattening N along all indices
	other than that associated with t2[index], and doing the same for W.

	Note that this method requires that norm(t1) == norm(t2) == 1.
	'''

	# Contract the connections between the two lists
	cont = [np.tensordot(a,b, axes=((1,1))) for a,b in zip(*(t1, t2))]
	
	# Rotate cont so that index becomes len(cont)-1.
	cont = shift(cont, len(cont) - 1 - index)

	# Contract the two lists aside from index
	x = cont[0]
	x = np.swapaxes(x, 1, 2)
	for y in cont[1:len(cont)-1]:
		x = np.tensordot(x, y, axes=((2,3),(0,2)))


	# Now we construct N and W.

	# Contract with the tensor immediately opposing index. Flattening then yields W.
	W = np.tensordot(x, t1[index], axes=((2,0),(0,2)))
	W = np.swapaxes(W, 0, 1)
	W = np.swapaxes(W, 1, 2)
	W = np.reshape(W, (-1,))

	ret = t2[::]

	for _ in range(3):
		for i in range(len(ret)):
			print('IND:',i,index)
			op = normMat(t2, i)
			ret = t2[::]
			for _ in range(3):
				ret[i] = np.random.randn(*ret[i].shape)
				x = ret[i].reshape((-1,))
				print(norm(ret)-np.dot(x, np.dot(op, x)))
				print(norm(ret)-np.dot(x, normOp(t2, i).matvec(x)))

	# Something breaks here which causes x.op.x to not always be norm(ret).
	# Whatver it is is persistent (e.g. breaks in the same place even when op
	# is regenerated), does not rely on modifying ret or t2, and generally
	# only affects op for some indices and not others.
	# In addition, for unclear reasons it only ever affects non-terminal indices (e.g.
	# not the final one in t2). In fact, it seems to only affect the first two indices
	# (eg t2[0] and t2[1]). This points to it being associated with the construction of the matrix
	# rather than subsequent logic.
	# 
	# The fact that also affects the operator form in the same indices but with different results suggests an
	# underlying logic error.

	op = normMat(t2, index)
	res = np.linalg.solve(op, W).reshape(t2[index].shape)
#	res = lsqr(op, W)[0].reshape(t2[index].shape)

	ret = t2[::]
	ret[index] = res

	x = ret[index].reshape((-1,))
	y = t2[index].reshape((-1,))
	err0 = norm(t1) + np.dot(y, np.dot(op, y)) - 2 * np.dot(y, W)
	err1 = norm(t1) + np.dot(x, np.dot(op, x)) - 2 * np.dot(x, W)

	print('CCC',contract(t1,ret) - np.dot(x,W))
	print('BBB',norm(ret) / np.dot(x, np.dot(op, x)), norm(ret), np.sum(np.abs(np.dot(op, ret[index].reshape((-1,))) - W)))
	print('DDD',np.sum(x**2))

	print(err0, err1, np.sum(np.abs(op-op.T)), norm(ret) - np.dot(x, np.dot(op, x)), norm(t1), norm(ret), np.dot(x,W))
	assert abs(norm(ret) / np.dot(x, np.dot(op, x)) - 1) < 1e-3
	assert err0 >= 0
	assert err1 >= 0
#	assert err1 <= err0

	return ret, err0, err1

def optimizeRank(tensors, ranks, stop=1e-2, start=None):
	'''
	tensors is a list of rank-3 tensors set such that the last index of each contracts
	with the first index of the next, and the last index of the last tensor contracts
	with the first index of the first one.

	The return value is an optimized list of tensors representing the original as best
	as possible subject to the constraint that the bond dimensions of the returned tensors
	are given by the entries in ranks, the first of which gives the dimension between the
	first and second tensors. The current approximation error is also returned.

	The ending criterion is given by the parameter stop, which specifies the minimum logarithmic
	change in the error between optimization sweeps (once it drops below this point the algorithm halts).

	A starting point for optimization may be provided using the option start.
	'''
	
	# Normalize tensors
#	n = norm(tensors)
#	tensors[0] /= np.sqrt(n)

	# Generate random starting point and normalize.
	if start is not None:
		t2 = start[::]	
	else:
		t2 = [np.random.rand(ranks[i-1],t.shape[1],ranks[i]) for i,t in enumerate(tensors)]
		t2[0] /= np.sqrt(norm(t2))

	# Optimization loop
	dlnerr = 1

	while dlnerr > 0.01:
		for i in range(len(tensors)):
			t2, err1, err2 = optimizeTensor(tensors, t2, i)
			derr = (err1 - err2)
			dlnerr = derr / err1
			print(dlnerr,err1,err2)
			err1 = err2

	print('Done.')
	# Restore normalization
#	t2[0] *= np.sqrt(n)
#	tensors[0] *= np.sqrt(n)

	return t2, err1


def optimize(tensors, tol):
	'''
	tensors is a list of rank-3 tensors set such that the last index of each contracts
	with the first index of the next, and the last index of the last tensor contracts
	with the first index of the first one.

	The return value is an optimized list of tensors representing the original to
	relative L2 error tol.
	'''

	# Initialize ranks to one.
	ranks = [1 for _ in range(len(tensors))]

	# Optimization loop
	t2, err = optimizeRank(tensors, ranks)
	while err > tol:		
		options = []
		print(err)
		# Try different rank increases
		for i in range(len(tensors)):
			ranksNew = ranks[::]
			ranksNew[i] += 1

			print(ranksNew)

			# Generate starting point
			start = t2[::]

			# Enlarge tensor to the left of the bond
			sh = list(start[i].shape)
			sh[2] += 1
			start[i] = np.zeros(sh)
			start[i][:,:,:-1] = t2[i]

			# Enlarge tensor to the right of the bond
			sh = list(start[(i+1)%len(start)].shape)
			sh[0] += 1
			start[(i+1)%len(start)] = np.zeros(sh)
			start[(i+1)%len(start)][:-1,:,:] = t2[(i+1)%len(start)]

			# Optimize
			t2New, errNew = optimizeRank(tensors, ranksNew, start=None)
			options.append((ranksNew, t2New, errNew))
			print(errNew)

		# Pick the best option
		print(min(options, key=lambda x: x[2])[2], err)
		assert min(options, key=lambda x: x[2])[2] < err
		ranks, t2, err = min(options, key=lambda x: x[2])
		print(ranks, err)

	return ranks, t2, err

def kronecker(dim):
	x = np.zeros((dim, dim, dim))
	for i in range(dim):
		x[i,i,i] = 1
	return x

def test(dim):
	x = 1 + np.random.randn(dim,dim,dim)
	return x

tensors = [test(2) for _ in range(5)]
tensors[0] /= np.sqrt(norm(tensors))
optimize(tensors, 1e-5)
	
