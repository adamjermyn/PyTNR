import numpy as np
from copy import deepcopy

from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TensorLoopOptimization.optTensor import optTensor


def norm(t):
    '''
    Returns the L2 norm of the tensor.
    '''
    t1 = t.copy()
    t2 = t.copy()

    return t1.contract(range(t.rank), t2, range(t.rank), elimLoops=False).array

def cost(tensors):
	return tensors.compressedSize

class optimizer:
	def __init__(self, tensors, tolerance, cut=False):

		# Normalize tensors
		self.norm = np.sqrt(norm(tensors))
		temp = tensors.externalBuckets[0].node.tensor.array
		temp /= self.norm
		tensors.externalBuckets[0].node.tensor = ArrayTensor(temp)

		# Reorder externalBuckets to match the underlying ordering of nodes in the loop.
		# At the end of the day what we'll do is read out the tensors in the loop by
		# bucket index and slot them back into the full network in place of the
		# original tensors with the same bucket index. As a result we can feel free
		# to move around the buckets in externalBuckets so long as the buckets themselves
		# are unchanged. The key is that we never need to use the loop network
		# (which is a NetworkTensor) as a tensor proper.

		self.inds = list([0 for _ in range(tensors.rank)])
		buckets = [tensors.externalBuckets[0]]
		n = buckets[0].node
		prevNodes = set()
		prevNodes.add(n)
		while len(buckets) < len(tensors.externalBuckets):
			n = tensors.network.internalConnected(n).difference(prevNodes).pop()
			exb = list(b for b in n.buckets if b in tensors.externalBuckets)[0]
			buckets.append(exb)
			self.inds[tensors.externalBuckets.index(exb)] = len(prevNodes)
			prevNodes.add(n)
		tensors.externalBuckets = buckets

		# Store inputs
		self.tensors = tensors
		self.tolerance = tolerance
		self.cut = cut

		# Store past attempts
		self.active = []
		self.stored = {}
		self.nanCount = 0

		# First evaluation
		x = optTensor(self.tensors)
		x.optimizeSweep()
		t = deepcopy(x.guess)
		err = x.error
		derr = 1 - err
		c = cost(t)
		dc = c
		self.stored[x] = (t, err, derr, c, dc)
		self.active.append(x)

	@property
	def len(self):
	    return len(self.tensors)	

	def choose(self):
		return self.active[-1]

	def makeNext(self):
		# Choose an option to improve
		previous = self.choose()

		for i in range(len(previous)):
			if not cut or len(list(1 for x in previous.ranks if x == 1)) > 1 or previous.ranks[i] != 1:
				new = deepcopy(previous)
				new.expand(i)
				new.optimizeSweep()
				err = new.error
				t = new.guess

				print(new, new.fullError, err, norm(t), self.norm)
				if new.fullError < self.tolerance:
					temp = t.externalBuckets[0].node.tensor.array
					temp *= self.norm
					t.externalBuckets[0].node.tensor = ArrayTensor(temp)
					print(norm(t))
					return t, err

				if np.isnan(err):# or err > 1.1 * self.stored[previous][1]:
					self.stored[new] = (None, 1e100, 0, 0, 0)
					self.nanCount += 1
				else:
					self.active.append(new)
					derr = self.stored[previous][1] - err
					dc = cost(t) - self.stored[previous][3]
					c = cost(t)
					self.stored[new] = (t, err, derr, c, dc)

		self.active.remove(previous)
		self.active = sorted(self.active, key=lambda x: self.stored[x][1], reverse=True)
		if self.stored[self.active[-1]][1] > self.tolerance:
			return None

def cut(tensors, tolerance):
	opt = optimizer(tensors, tolerance, cut=True)
	ret = None
	while ret is None:
		ret = opt.makeNext()
		if ret is not None:
			ret, err = ret
#		if opt.nanCount > 20:
#			opt = optimizer(tensors, tolerance, cut=True)
	return ret, opt.inds, err



#def kronecker(dim):
#   x = np.zeros((dim, dim, dim))
#   for i in range(dim):
#       x[i,i,i] = 1
#   return x

#def test(dim):
#   x = 5 + np.random.randn(dim,dim,dim)
#   return x

#def testTensors(dim, num):
    # Put no correlations in while maintaining a non-trivial physical index
#   tens = np.random.randn(dim, dim, dim)
#   for i in range(dim-1):
#       tens[:,i,:] = np.random.randn()

    # Apply random unitary operators.
#   from scipy.stats import ortho_group
#   tensors = [np.copy(tens) for _ in range(num)]
#   for i in range(num):
#       u = ortho_group.rvs(dim)
#       uinv = np.linalg.inv(u)
#       tensors[i] = np.einsum('ijk,kl->ijl', tensors[i], u)
#       tensors[(i+1)%num] = np.einsum('li,ijk->ljk', uinv, tensors[i])

#   return tensors

#def test2(dimRoot):
    # Constructs a tensor list which, upon contraction, returns 0 when
    # all indices are zero and 1 otherwise.

#   dim = dimRoot**2

#   t = np.ones((dim,dim,dim))
#   t[0,0,0] = 0

    # SVD
#   t = np.reshape(t, (dim**2, dim))
#   u, s, v = np.linalg.svd(t, full_matrices=False)
#   q = np.dot(u, np.diag(np.sqrt(s)))
#   r = np.dot(np.diag(np.sqrt(s)), v)

    # Insert random unitary between q and r
#   from scipy.stats import ortho_group
#   u = ortho_group.rvs(dim)
#   uinv = np.linalg.inv(u)
#   q = np.dot(q, u)
#   r = np.dot(uinv, r)

    # Decompose bond
#   q = np.reshape(q,(dim, dim, dimRoot, dimRoot))
#   r = np.reshape(r,(dimRoot, dimRoot, dim))

    # Split q
#   q = np.swapaxes(q, 1, 2)

#   q = np.reshape(q, (dim*dimRoot, dim*dimRoot))
#   u, s, v = np.linalg.svd(q, full_matrices=False)

#   a = np.dot(u, np.diag(np.sqrt(s)))
#   b = np.dot(np.diag(np.sqrt(s)), v)

#   a = np.reshape(a, (dim, dimRoot, dim*dimRoot))
#   b = np.reshape(b, (dim*dimRoot, dim, dimRoot))

#   print(np.einsum('ijk,kml,jlw->imw',a,b,r))


#   return [a,b,r]

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

