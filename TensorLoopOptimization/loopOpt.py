import numpy as np
from copy import deepcopy
from scipy.sparse.linalg import LinearOperator, bicgstab, lsqr

from TNR.Tensor.arrayTensor import ArrayTensor

def shift(l, n):
	'''
	Shifts the list l forward by n indices.
	'''

	n = n % len(l)
	return l[-n:] + l[:-n]

def contract(t1, t2):
	'''
	Contracts two NetworkTensors against one another along external buckets.
	The networks must have matching external bucket ID's.
	'''
	bids1 = list(b.id for b in t1.externalBuckets)
	bids2 = list(b.id for b in t2.externalBuckets)
	assert set(bids1) == set(bids2)

	inds1 = list(range(len(bids1)))
	inds2 = list(bids2.index(i) for i in bids1)

	t = t1.contract(inds1, t2, inds2)

	return t.array

def norm(t1):
	'''
	Returns the L2 norm of the specified NetworkTensor.
	'''

	return contract(t1, t1)

def cost(t):
	return t.compressedSize

def prepareNW(t1, t2, index):
	'''

	Following equation S10 in arXiv:1512.04938, we construct the operators N and W.

	N is given by removing the node at index from t2 and contracting it against itself.
	W is given by removing the node at index from t2 and contracting it against t1.
	'''	

	# Copy t2 and remove the node at index
	t2new = deepcopy(t2)
	t2new.removeNode(t2new.externalBuckets[index].node)

	# Contract against t1 to generate W.
	w = contract(t1, t2new)

	# Contract t2 against itself to generate N.
	n = contract(t2new, t2new)

	# Bolt on the identity
	iden = np.identity(t2.externalBuckets[index].size)
	n.addTensor(ArrayTensor(iden))

	# Construct array versions of W and N
	arrW = w.array
	arrN = n.array

	# Align indices in N to agree with those in W.
	perm = list(0 for _ in range(6))
	for j,b in enumerate(w.externalBuckets):
		# Corresponding buckets in t1 and t2 have the same ID's.
		indices = list(i for i,bb in enumerate(n.externalBuckets) if bb.id==b.id)
		perm[j] = indices[0]
		perm[j + 3] = indices[1]

	arrN = np.transpose(arrN, axes=perm)

	return arrN, arrW, n, w

def optimizeTensor(t1, t2, index):
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

	# Now we construct N and W.

	N, W, n, w = prepareNW(t1, t2)

	# Reshape into matrices
	W = np.reshape(W, (-1,))
	N = np.reshape(N, (len(W), len(W)))

	try:
		res = np.linalg.solve(N, W)
	except np.linalg.linalg.LinAlgError:
		res = lsqr(N, W)[0]

	ret = deepcopy(t2)

	# Permute axes, un-flatten and put res into ret at the appropriate place.

	perm = []
	for b in t2.externalBuckets[index].node.buckets:
		ind = list(bb.id for bb in w.externalBuckets).index(b.id)
		perm.append(ind)

	res = np.transpose(res, axes=perm)
	res = np.reshape(res, t2.externalBuckets[index].node.tensor.shape)
	ret.externalBuckets[index].node.tensor = res

	err = 2 * (1 - contract(t1, ret))

	return ret, err

def optimizeRank(tensors, ranks, stop=0.1, start=None):
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
		err1 = err2

	return t2, err1

def kronecker(dim):
	x = np.zeros((dim, dim, dim))
	for i in range(dim):
		x[i,i,i] = 1
	return x

def test(dim):
	x = 5 + np.random.randn(dim,dim,dim)
	return x

def testTensors(dim, num):
	# Put no correlations in while maintaining a non-trivial physical index
	tens = np.random.randn(dim, dim, dim)
	for i in range(dim-1):
		tens[:,i,:] = np.random.randn()

	# Apply random unitary operators.
	from scipy.stats import ortho_group
	tensors = [np.copy(tens) for _ in range(num)]
	for i in range(num):
		u = ortho_group.rvs(dim)
		uinv = np.linalg.inv(u)
		tensors[i] = np.einsum('ijk,kl->ijl', tensors[i], u)
		tensors[(i+1)%num] = np.einsum('li,ijk->ljk', uinv, tensors[i])

	return tensors

def test2(dimRoot):
	# Constructs a tensor list which, upon contraction, returns 0 when
	# all indices are zero and 1 otherwise.

	dim = dimRoot**2

	t = np.ones((dim,dim,dim))
	t[0,0,0] = 0

	# SVD
	t = np.reshape(t, (dim**2, dim))
	u, s, v = np.linalg.svd(t, full_matrices=False)
	q = np.dot(u, np.diag(np.sqrt(s)))
	r = np.dot(np.diag(np.sqrt(s)), v)

	# Insert random unitary between q and r
	from scipy.stats import ortho_group
	u = ortho_group.rvs(dim)
	uinv = np.linalg.inv(u)
	q = np.dot(q, u)
	r = np.dot(uinv, r)

	# Decompose bond
	q = np.reshape(q,(dim, dim, dimRoot, dimRoot))
	r = np.reshape(r,(dimRoot, dimRoot, dim))

	# Split q
	q = np.swapaxes(q, 1, 2)

	q = np.reshape(q, (dim*dimRoot, dim*dimRoot))
	u, s, v = np.linalg.svd(q, full_matrices=False)

	a = np.dot(u, np.diag(np.sqrt(s)))
	b = np.dot(np.diag(np.sqrt(s)), v)

	a = np.reshape(a, (dim, dimRoot, dim*dimRoot))
	b = np.reshape(b, (dim*dimRoot, dim, dimRoot))

	print(np.einsum('ijk,kml,jlw->imw',a,b,r))


	return [a,b,r]

#dimRoot = 5

#dim = dimRoot**2
#tensors = test2(dimRoot)

#a,b,r = tensors

#print(np.linalg.svd(np.einsum('ijk,kml->ijml',a,b).reshape((dim*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',b,r).reshape((dim**2*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',r,a).reshape((dimRoot**2,dim*dimRoot**2)))[1])

#tensors = [test(5) for _ in range(5)]
#tensors = testTensors(5, 5)
#tensors[0] /= np.sqrt(norm(tensors))
#print(optimize(tensors, 1e-5)[:2])
	

#print(np.linalg.svd(np.einsum('ijk,kml->ijml',a,b).reshape((dim*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',b,r).reshape((dim**2*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',r,a).reshape((dimRoot**2,dim*dimRoot**2)))[1])

