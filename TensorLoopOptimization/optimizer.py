import numpy as np

from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TensorLoopOptimization.loopOpt import optimizeRank, norm, expand

def cost(tensors):
	return sum(t.size for t in tensors)

def next(previous, cut):
	options = []

	inds = list(range(len(previous)))

	if cut and len([1 for r in previous if r == 1]) == 1:
		inds.remove(previous.index(1))

	for i in inds:
		r = list(previous)
		r[i] += 1
		options.append((i, tuple(r)))

	return options

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

		buckets = [tensors.externalBuckets[0]]
		n = buckets[0].node
		while len(buckets) < len(tensors.externalBuckets):
			prevN = n
			n = tensors.network.internalConnected(n).difference(set(prevN)).pop()
			exb = list(b for b in n.buckets if b in tensors.externalBuckets)[0]
			buckets.append(exb)
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
		x = tuple(1 for _ in range(len(tensors)))
		t, err = optimizeRank(tensors, x)
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
		options = next(previous, self.cut)

		for i,n in options:
			# Get starting point from previous
			t = self.stored[previous][0]

			# Enlarge tensor to the left of the bond
			start = expand(t, i)

			# Optimize
			t, err = optimizeRank(self.tensors, n, start)
			print(n, self.stored[previous][1], err)
			if err < self.tolerance:
				temp = t.externalBuckets[0].node.tensor.array
				temp *= self.norm
				t.externalBuckets[0].node.tensor = ArrayTensor(temp)
				return t

			if np.isnan(err) or err > 1.1 * self.stored[previous][1]:
				self.stored[n] = (None, 1e100, 0, 0, 0)
				self.nanCount += 1
			else:
				self.active.append(n)
				derr = self.stored[previous][1] - err
				dc = cost(t) - self.stored[previous][3]
				c = cost(t)
				self.stored[n] = (t, err, derr, c, dc)

		self.active.remove(previous)
		self.active = sorted(self.active, key=lambda x: self.stored[x][1], reverse=True)
		if self.stored[self.active[-1]][1] > self.tolerance:
			return None

def cut(tensors, tolerance):
	opt = optimizer(tensors, tolerance, cut=True)
	ret = None
	while ret is None:
		ret = opt.makeNext()
		if opt.nanCount > 20:
			opt = optimizer(tensors, tolerance, cut=True)
	return ret