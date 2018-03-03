import numpy as np

from TNR.TensorLoopOptimization.loopOpt import optimizeRank, norm

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
		tensors = list(np.array(t) for t in tensors)
		tensors[0] /= self.norm
		self.nanCount = 0

		self.tensors = tensors
		self.tolerance = tolerance
		self.cut = cut

		# Store past attempts
		self.active = []
		self.stored = {}

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
			start = t[::]

			# Enlarge tensor to the left of the bond
			sh = list(start[i])
			sh = list(start[i].shape)
			sh[2] += 1
			start[i] = np.random.randn(*sh)
			start[i][:,:,:-1] = t[i]

			# Enlarge tensor to the right of the bond
			sh = list(start[(i+1)%len(start)].shape)
			sh[0] += 1
			start[(i+1)%len(start)] = np.random.randn(*sh)
			start[(i+1)%len(start)][:-1,:,:] = t[(i+1)%len(start)]

			# Optimize
			t, err = optimizeRank(self.tensors, n, start=start)
			if err < self.tolerance:
				t[0] *= self.norm
				return t

			print(n, self.stored[previous][1], err)
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