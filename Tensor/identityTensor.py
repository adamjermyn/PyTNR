import numpy as np
from link import Link
from treeTensor import TreeTensor
from arrayTensor import ArrayTensor
from collections import defaultdict
def layer(n):
	return int(np.log2(n/3)) + 2

class IdentityTensor(TreeTensor):
	'''
	This is a special class for constructing the rank-n identity tensor.
	This is done in a tree from the start to support large n.
	'''


	def __init__(self, dimension, rank):
		super().__init__(0.0)

		numLayers = layer(dimension)

		assert rank >= 2

		if rank == 2:
			return ArrayTensor(np.identity(dimension))

		numTensors = rank - 2

		buckets = []

		# Create identity array
		iden = np.zeros((dimension,dimension,dimension))
		for i in range(dimension):
			iden[i,i,i] = 1.0

		for i in range(numTensors):
			n = super().addTensor(ArrayTensor(iden))
			buckets = buckets + n.buckets


		while len(self.network.externalBuckets) > rank:
			b = buckets.pop(0)
			i = 0
			while buckets[i].node is b.node or len(buckets[i].node.connectedNodes) > 0:
				i += 1
			Link(b, buckets[i])

			self.externalBuckets.remove(b)
			self.externalBuckets.remove(buckets[i])
			self.network.externalBuckets.remove(b)
			self.network.externalBuckets.remove(buckets[i])
			self.network.internalBuckets.add(b)
			self.network.internalBuckets.add(buckets[i])

			buckets.remove(buckets[i])
