import numpy as np
from copy import deepcopy

from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TensorLoopOptimization.optTensor import optTensor

from TNR.Utilities.graphPlotter import plot

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])


'''
Todo:

It may be faster to cut using SVD.

Select a bond to cut. Tensor product all other tensors with the identity
to replace that bond. Then pass back and forth with SVD (starting on one end or the other
because the middle will not compress) until the SVD stops haivng an impact. Do this for
each possible bond to cut and pick the one with the least cost.

The SVD is probably better in part because it can be done specifically to the accuracy
of interest rather than having to go back and forth until that's reached. That is, it removes
the searching for optimal ranks. It can readily incorporate the environment too.

Note that it is not faster to optimize by SVD: this will not catch the loop entropy.

Further note that there is a problem with cutting by SVD: It will initially significantly
overestimate the memory cost of the cut because the tensor product will produce the largest
bonds which could possibly be required, and only after O(N) svd operations will these be
significantly reduced (e.g. once the changes propagate along the identity line to the other side).

A possible compromise is therefore to use the least squares solver in the expanding phase
and SVD in the reduction phase. Most of the time is spent in the reduction phase anyway
because that scales exponentially in loop size, so this is not a bad option.


Also, should put a cap on how far into the network we go to compute the environment.
This is just a polynomial scaling problem but becomes significant with large networks.

'''


def norm(t):
	'''
	The L2 norm of the tensor. Must be a NetworkTensor.
	'''
	t1 = t.copy()
	t2 = t.copy()

	return t1.contract(range(t.rank), t2, range(t.rank), elimLoops=False).array

def envNorm(t, env):
	'''
	The L2 norm of the tensor contracted with its environment. Must be a NetworkTensor.
	'''
	t1 = t.copy()
	t2 = t.copy()

	c = t1.contract(range(t.rank), env, range(t.rank), elimLoops=False)
	c = c.contract(range(t.rank), t2, range(t.rank), elimLoops=False)

	return c.array

def densityMatrix(loop, env, indices):
	'''
	The density matrix formed by the NetworkTensor specified by loop contracted
	against itself and the environment NetworkTensor env on all but the specified indices. 
	'''
	t1 = loop.copy()
	t2 = loop.copy()
	inds = list(set(range(t1.rank)).difference(set(indices)))
	c = t1.contract(range(t1.rank), env, range(t1.rank), elimLoops=False)
	c = c.contract(inds, t2, inds, elimLoops=False)
	return c.array

def between(i,j,ind,cutInd):
	'''
	In a loop with nodes 0...N-1, cut the link between nodes cutInd and cutInd+1.
	Returns True if ind lies between i and j and False otherwise.
	Requires i != j.
	'''
	assert i != j

	if i > j:
		i,j = j,i

	if i <= cutInd and cutInd < j:
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
	def __init__(self, tensors, tolerance, environment, bids, otherbids):

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

		# Store past attempts
		self.active = []
		self.stored = {}
		self.nanCount = 0

		# Figure out which link to cut
		acc = tolerance / tensors.rank
		dims = {}

		for i in range(tensors.rank):
			for j in range(i):
				rho = densityMatrix(self.tensors, self.environment, (i,j)) # Change to use environment
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

		# Control flow
		self.status = 'Expanding'

		# First evaluation
		print(best[2])
		x = optTensor(self.tensors, self.environment)
		for i in range(len(best[2])):
			for j in range(best[2][i]-1):
				x.expand(i)

		x.optimizeSweep(self.tolerance)
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


		if self.status == 'Expanding':
			print('Status: Expanding')
			# We haven't found something in our error criterion yet, so expand.			
			new = deepcopy(previous)
			for i in range(len(previous)):
				if len(list(1 for x in previous.ranks if x == 1)) > 1 or previous.ranks[i] != 1:
					new.expand(i)
			
			new.optimizeSweep(self.tolerance)
		
			err = new.error
			t = new.guess

			if err < self.tolerance:
				# The error criterion is satisfied, so now we reduce.
				self.status = 'Decreasing'
				print('Status: Decreasing')
		elif self.status == 'Decreasing':
			found = False
			for i in range(len(previous)):
				if previous.ranks[i] > 1:
					new = deepcopy(previous)
					new.reduce(i)
					print(new.ranks)
					new.optimizeSweep(self.tolerance)
					err = new.error
					t = new.guess
					if err < self.tolerance:
						# Found a reduction which doesn't break the error criterion.
						found = True
						break
			if not found:
				# No reductions found. Proceed to return.
				t = previous.guess
				err = previous.error
				temp = t.externalBuckets[0].node.tensor.array
				temp *= self.norm
				t.externalBuckets[0].node.tensor = ArrayTensor(temp)
				return t, err


		# Either status is expanding or status is decreasing and a reduction was found.
		# In either case we have a new valid solution so we write it in.
		print(new, previous, err, self.tolerance, envNorm(t, self.environment))
		if np.isnan(err):# or err > 1.1 * self.stored[previous][1]:
			self.stored[new] = (None, 1e100, 0, 0, 0)
			self.nanCount += 1
		else:
			self.active.append(new)
			derr = self.stored[previous][1] - err
			c = max(new.ranks)
			dc = 0
			self.stored[new] = (t, err, derr, c, dc)

		self.active.remove(previous)
		self.active = sorted(self.active, key=lambda x: self.stored[x][1] + np.log(self.stored[x][3]), reverse=True)
		if self.stored[self.active[-1]][1] > self.tolerance:
			return None

def cut(tensors, tolerance, environment, bids, otherbids):
	opt = optimizer(tensors, tolerance, environment, bids, otherbids)
	ret = None
	while ret is None:
		ret = opt.makeNext()
		if ret is not None:
			ret, err = ret
	return ret, opt.inds, err



