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
	def __init__(self, tensors, tolerance, environment, bids, otherbids, ranks, lids):

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

		# Construct guess
		x = optTensor(self.tensors, self.environment)
		for i in range(len(self.inds)):
			# Identify ranks by link ID's
			ID = self.tensors.externalBuckets[i].node.findLink(self.tensors.externalBuckets[(i+1)%self.tensors.rank].node).id
			ind = lids.index(ID)
			x.expand(i, amount=int(ranks[ind] - 1))

		# Control flow
		x.optimizeSweep(self.tolerance)
		
		# Undo normalization
		temp = x.guess.externalBuckets[0].node.tensor.array
		temp *= self.norm
		x.guess.externalBuckets[0].node.tensor = ArrayTensor(temp)
		
		self.x = x

def cut(tensors, tolerance, environment, bids, otherbids, ranks, lids):
	opt = optimizer(tensors, tolerance, environment, bids, otherbids, ranks, lids)
	ret = opt.x.guess
	return ret, opt.inds



