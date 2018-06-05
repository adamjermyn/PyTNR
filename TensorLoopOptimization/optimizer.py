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

def envNorm(t, env):
	t1 = t.copy()
	t2 = t.copy()

	c = t1.contract(range(t.rank), env, range(t.rank), elimLoops=False)
	c = c.contract(range(t.rank), t2, range(t.rank), elimLoops=False)

	return c.array

def cost(tensors):
	return max(n.tensor.size for n in tensors.network.nodes)

def densityMatrix(loop, indices):
	t1 = loop.copy()
	t2 = loop.copy()
	inds = list(set(range(t1.rank)).difference(set(indices)))
	c = t1.contract(inds, t2, inds, elimLoops=False)
	return c.array

def between(i,j,ind,cut):
    # True if ind lies between i and j after the bond at cut has been removed.
    # Assumes i < j.
    if i <= cut and cut < j:
        if ind >= j or ind < i:
            return True
        else:
            return False
    else:
        if ind >= j or ind < i:
            return False
        else:
            return True

class optimizer:
	def __init__(self, tensors, tolerance, environment, bids, otherbids, cut=False):

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

		# Contract the environment against itself
		inds = []
		for i in range(environment.rank):
			if environment.externalBuckets[i].id not in otherbids:
				inds.append(i)

		newEnv = environment.copy()
		environment = environment.contract(inds, newEnv, inds, elimLoops=False)
		environment.contractRank2()

		# Normalise environment
		envn = norm(environment)
		environment.externalBuckets[0].node.tensor = ArrayTensor(environment.externalBuckets[0].node.tensor.array / np.sqrt(envn))

		# There are two external buckets for each external on loop (one in, one out),
		# and these are now ordered in two sets which correspond, as in
		# [p,q,r,..., p,q,r,...]
		# Now we reorder these to match the order of nodes in the loop.
		buckets1 = []
		buckets2 = []
		extbids = list(b.id for b in environment.externalBuckets)
		for b in tensors.externalBuckets:
			ind = bids.index(b.id)

			other = otherbids[ind]
			ind = extbids.index(other)
			buckets1.append(environment.externalBuckets[ind])
			buckets2.append(environment.externalBuckets[ind + tensors.rank])

		environment.externalBuckets = buckets1 + buckets2

		# Normalize tensors
		self.norm = np.sqrt(envNorm(tensors, environment))
		temp = tensors.externalBuckets[0].node.tensor.array
		temp /= self.norm
		tensors.externalBuckets[0].node.tensor = ArrayTensor(temp)

		# Store inputs
		self.tensors = tensors
		self.environment = environment
		self.tolerance = tolerance
		self.cut = cut

		# Store past attempts
		self.active = []
		self.stored = {}
		self.nanCount = 0

		# Figure out which link to cut

		acc = tolerance
		dims = {}

		for i in range(tensors.rank):
			for j in range(i):
				rho = densityMatrix(self.tensors, (i,j)) # Change to use environment
				rho = np.reshape(rho, (rho.shape[0]*rho.shape[1], rho.shape[0]*rho.shape[1]))
				u, s, v = np.linalg.svd(rho)
				s /= np.sum(s)
				p = np.cumsum(s)
				p = 1 - p
				p = p[::-1]
				dims[(i,j)] = len(p) - np.searchsorted(p,acc)
				dims[(j,i)] = len(p) - np.searchsorted(p,acc)

		best = [1e100, 1e100, None, None, None]
		for i in range(tensors.rank):
			# So we cut the bond to the right of i
			mins = [1 for _ in range(tensors.rank)]
			maxs = [0 for _ in range(tensors.rank)]
			for j in range(tensors.rank):
				for k in range(j):
					for q in range(len(mins)):
						if between(k,j,q,i):
							if mins[q] < dims[(j,k)]:
								mins[q] = dims[(j,k)]
							maxs[q] += dims[(j,k)]
			if sum(mins) < best[0]:
				best = [sum(mins), sum(maxs), mins, maxs, i]
			elif sum(mins) == best[0] and sum(maxs) < best[1]:
				best = [sum(mins), sum(maxs), mins, maxs, i]

		# First evaluation
		print(best[2])
		x = optTensor(self.tensors, self.environment)
		for i in range(len(best[2])):
			for j in range(best[2][i]-1):
				x.expand(i)

		x.optimizeSweep()
		t = deepcopy(x.guess)
		err = x.error
		derr = 1 - err
		c = 1
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

#				print(new, new.fullError, err, norm(t), self.norm)
#				if new.fullError < self.tolerance:
				print(new, previous, err, envNorm(t, self.environment))
				if err < self.tolerance:
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
					c = max(new.ranks)
					dc = 0
#					dc = cost(t) - self.stored[previous][3]
#					c = cost(t)
					self.stored[new] = (t, err, derr, c, dc)

		self.active.remove(previous)
		self.active = sorted(self.active, key=lambda x: self.stored[x][1] + np.log(self.stored[x][3]), reverse=True)
		if self.stored[self.active[-1]][1] > self.tolerance:
			return None

def cut(tensors, tolerance, environment, bids, otherbids):
	opt = optimizer(tensors, tolerance, environment, bids, otherbids, cut=True)
	ret = None
	while ret is None:
		ret = opt.makeNext()
		if ret is not None:
			ret, err = ret
#		if opt.nanCount > 20:
#			opt = optimizer(tensors, tolerance, cut=True)
	return ret, opt.inds, err



