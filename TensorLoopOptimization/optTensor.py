from copy import deepcopy

def norm(t):
	t1 = t.copy()
	t2 = t.copy()

	return t1.contract(range(t.rank), t2, range(t.rank)).array

class optTensor:
	def __init__(self, loop, guess):
		self.loop = loop
		self.guess = guess

	@property
	def loopNorm(self):
		return norm(self.loop)

	@property
	def guessNorm(self):
		return norm(self.guess)

	@property
	def error(self):
		t1 = self.loop.copy()
		t2 = self.guess.copy()
		c = t1.contract(range(t1.rank), t2, range(t1.rank)).array
		return 2*(1 - c)

	def oneRemoved(self, index):
		# Returns the contraction of loop with guess with the node
		# associated with the specified external index of guess removed.

		t1 = deepcopy(self.loop)
		t2 = deepcopy(self.guess)
		t2.removeNode(t2.externalBuckets[index].node)

		ind1 = list(range(t1.rank))
		ind1.remove(index)

		# Three internal buckets have become external, and these are at the end.
		ind2 = list(range(t2.rank) - 3)

		c = t1.contract(ind1, t2, ind2)
		arr = c.array

		# Because we deep-copied, these bucket ID's are as they were in self.guess.
		bids = list(b.id for b in c.externalBuckets)

		return arr, bids

	def bothRemoved(self, index):
		# Returns the contraction of loop with guess, with the nodes
		# associated with the specified external index of both removed.
		# Also inserts the identity between the removed nodes.

		