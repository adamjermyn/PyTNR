from network import Network
from node import Node
from bucket import Bucket
from link import Link
from arrayTensor import ArrayTensor
from utils import entropy, split_chunks, splitArray

class treeNetwork(Network):
	'''
	A treeNetwork is a special case of a Network in which the Network being represented
	contains no cycles. This allows matrix elements of a treeNetwork to be efficiently
	evaluated.

	As the only quantities which matter are the matrix elements, the treeNetwork may
	refactor itself through singular value decomposition (SVD) to minimize memory use, and
	so no assumptions should be made about the Nodes in this object, just the external
	Buckets.

	Internally all Nodes of a treeNetwork have rank-3 Tensors. SVD factoring is used
	to enforce this.
	'''

	def __init__(self, accuracy=1e-4):
		'''
		treeNetworks require an accuracy argument which determines how accurately (in terms of relative error)
		they promise to represent their matrix elements.
		'''
		super().__init__()

		self.accuracy = accuracy

		self.externalBuckets = []

	def pathBetween(self, node1, node2, calledFrom=None):
		'''
		Returns the unique path between node1 and node2.
		This is done by treating node1 as the root of the binary tree and performing a depth-first search.
		'''
		if node1 == node2: # Found it!
			return [node1]

		if len(node1.connected()) == 1 and calledFrom is not None:	# Nothing left to search
			return []

		for c in node1.connected(): # Search children
			if c is not calledFrom:
				path2 = self.pathBetween(c, node2, calledFrom=node1)
				if len(path2) > 0:
					return [node1] + path2

		return []

	def splitNode(self, node, prevIndex=None):
		'''
		Takes as input a Node and ensures that it is at most rank 3 by splitting it recursively.
		Any indices listed in prevIndex will be ignored in balancing the split, so this may be
		used to enforce an ordering of indices.
		'''
		if len(node.tensor.shape) > 3:
			self.deregisterNode(node)

			array = node.tensor.array

			if prevIndex is None:
				prevIndex = []

			s = []
			indices = list(range(len(array.shape)))
			for i in prevIndex:
				indices.remove(i)

			for i in indices:
				s.append((i, entropy(array, i)))

			chunks = split_chunks(s, 2)

			chunkIndices = [[i[0] for i in chunks[0]],[i[0] for i in chunks[1]]]

			u, v, prevIndices, indices1, indices2 = splitArray(array, [chunkIndices[0],chunkIndices[1]], ignoreIndex=prevIndex, accuracy=self.accuracy)

			b1 = Bucket()
			b2 = Bucket()
			n1 = Node(ArrayTensor(u), self, Buckets=[node.buckets[i] for i in indices1] + [b1])
			n2 = Node(ArrayTensor(v), self, Buckets=[b2] + [node.buckets[i] for i in indices2])
			l = Link(b1,b2)

			chunkIndices[1] = []
			for i in range(len(v.shape)):
				if i != 0 and i not in prevIndices:
					chunkIndices[1].append(i)

			if len(v.shape) <= 3:
				return [n2, self.splitNode(n1, prevIndex=[len(u.shape)-1])]

			self.deregisterNode(n2)
			q, v, prevIndices, indices1, indices2 = splitArray(v, [chunkIndices[1],[0]], ignoreIndex=[0] + prevIndices, accuracy=self.accuracy)
			b1 = Bucket()
			b2 = Bucket()
			n3 = Node(ArrayTensor(q), self, Buckets=[n2.buckets[i] for i in indices1] + [b1])
			n4 = Node(ArrayTensor(v), self, Buckets=[b2] + [n2.buckets[i] for i in indices2])
			l = Link(b1,b2)


			return [n4, self.splitNode(n1, prevIndex=[len(u.shape)-1]), self.splitNode(n3, prevIndex=[len(q.shape)-1])]
		else:
			return node
