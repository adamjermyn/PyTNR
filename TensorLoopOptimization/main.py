import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, lsqr

def shift(l, n):
	'''
	Shifts the list l forward by n indices.
	'''

	n = n % len(l)
	return l[-n:] + l[:-n]

def zipTensors(t1, t2):
	'''
	t1 and t2 are lists of equal length of rank-3 tensors. The second dimension in each tensor
	in one list matches that of the corresponding tensor, such that they may be contracted.

	The return value is the list of rank-4 tensors which results from contracting corresponding
	tensors in these lists.
	'''
	return [np.tensordot(a,b, axes=((1,),(1,))) for a,b in zip(*(t1, t2))]

def contractRank4(tens1, tens2):
	'''
	tens1 and tens2 are rank-4 tensors resulting from a zipTensors operation.
	It is assumed that they are each wired in the following way:

		0 -		  - 1
			Tensor
		2 - 	  - 3

	The return value is their contraction (placed side by side in the orientation shown above).
	This is given such that the indices not involved in the contraction are numbered in the same fashion
	as before the contraction.
	'''

	# Contract tensors
	x = np.tensordot(tens1, tens2, axes=((1,3),(0,2)))

	# Now the result has the form
	#
	#	0 -		  - 2
	#		Tensor
	#	1 - 	  - 3
	#
	# So we swap axes 1 and 2 to restore it to the desired form.
	x = np.swapaxes(x, 1, 2)

	return x

def contract(t1, t2):
	'''
	t1 and t2 are lists of rank-3 tensors set such that in each list the last index of
	each contracts with the first index of the next, and the last index of the last tensor
	contracts with the first index of the first one.

	The return value is the inner product of these tensors.
	'''

	# Contract the connections between the two lists
	cont = zipTensors(t1, t2)

	# Contract the two lists
	x = cont[0]
	for y in cont[1:]:
		x = contractRank4(x, y)

	# Contract the periodic indices
	n = np.einsum('iijj->',x)

	return n

def norm(tensors):
	'''
	tensors is a list of rank-3 tensors set such that the last index of each contracts
	with the first index of the next, and the last index of the last tensor contracts
	with the first index of the first one.

	The return value is the L2 norm of the tensor.
	'''
	return contract(tensors, tensors)

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
	cont = zipTensors(tensors, tensors)
	
	# Rotate cont so that index becomes len(cont)-1.
	cont = shift(cont, len(cont) - 1 - index)

	# Contract the two lists aside from index
	x = cont[0]
	for y in cont[1:len(cont)-1]:
		x = contractRank4(x, y)

	# Now the result has the form
	#
	#	0 -		  - 1
	#		Tensor
	#	2 - 	  - 3
	#
	# We outer product with the identity
	iden = np.identity(tensors[index].shape[1])
	x = np.tensordot(x, iden, axes=(tuple(),tuple()))
	# This gives
	#
	#	0 -		  - 1
	#		Tensor
	#	2 - 	  - 3
	#
	#	4 - Ident - 5
	#
	# Now we want to permute these indices so that upon flattening
	# the first three and the last three we obtain a matrix which
	# may be dotted against a vector formed by flattening a rank-3 tensor
	# to produce a result which may be un-flattened into a rank-3 tensor.
	# Indices 0, 1, and 5 dot against a rank-3 tensor, so these must become the last 3.
	# The identity (5) touches the middle index, so it must be the second in this group.
	# The first three must then be 2, 3, 4, with 4 in the middle.
	# Thus we want to permute the indices to be (2, 4, 3), (0, 5, 1).
	# This is not quite right, however, as the non-identity indices must match up with the convention
	# that the first in each group is the one that touches the left-index of the rank-3 object
	# and the final in each group is the one that touches the right-index. Thus we actually want
	# (3, 4, 2), (1, 5, 0).
	x = np.transpose(x, axes=(3,4,2,1,5,0))
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

	op = normMat(t2, index)
	res = np.linalg.solve(op, W).reshape(t2[index].shape)

	ret = t2[::]
	ret[index] = res

	x = ret[index].reshape((-1,))
	y = t2[index].reshape((-1,))
	err0 = norm(t1) + np.dot(y, np.dot(op, y)) - 2 * np.dot(y, W)
	err1 = norm(t1) + np.dot(x, np.dot(op, x)) - 2 * np.dot(x, W)

	return ret, err0, err1

def optimizeRank(tensors, ranks, stop, start=None):
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

	# Generate random starting point and normalize.
	if start is not None:
		t2 = start[::]	
	else:
		t2 = [np.random.rand(ranks[i-1],t.shape[1],ranks[i]) for i,t in enumerate(tensors)]
		t2[0] /= np.sqrt(norm(t2))

	# Optimization loop
	dlnerr = 1
	err1 = 1e100

	while dlnerr > stop:
		for i in range(len(tensors)):
			t2, _, err2 = optimizeTensor(tensors, t2, i)
		derr = (err1 - err2)
		dlnerr = derr / err1
		print(dlnerr,err1,err2)
		err1 = err2

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
	t2, err = optimizeRank(tensors, ranks, 1e-2)
	while err > tol:		
		options = []
		# Try different rank increases
		for i in range(len(tensors)):
			if ranks[i] < tensors[i].shape[1]:
				ranksNew = ranks[::]
				ranksNew[i] += 1

				# Generate starting point
				start = t2[::]

				# Enlarge tensor to the left of the bond
				sh = list(start[i].shape)
				sh[2] += 1
				start[i] = np.random.randn(*sh)
				start[i][:,:,:-1] = t2[i]

				# Enlarge tensor to the right of the bond
				sh = list(start[(i+1)%len(start)].shape)
				sh[0] += 1
				start[(i+1)%len(start)] = np.random.randn(*sh)
				start[(i+1)%len(start)][:-1,:,:] = t2[(i+1)%len(start)]

				# Optimize
				t2New, errNew = optimizeRank(tensors, ranksNew, err, start=start)
				options.append((ranksNew, t2New, errNew))
				print(ranksNew, errNew)

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
	x = 2 + np.random.randn(dim,dim,dim)
	return x

tensors = [test(5) for _ in range(5)]
tensors[0] /= np.sqrt(norm(tensors))
optimize(tensors, 1e-5)
	
