from network import Network

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

	def __init__(self):
		super.__init__()

		self.externalBuckets = []

	def pathBetween(self, node1, node2, calledFrom=None):
		'''
		Returns the unique path between node1 and node2.
		This is done by treating node1 as the root of the binary tree and performing a depth-first search.
		'''
		path = []

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

	def copy(self):
		'''
		Returns a copy of self.
		All Nodes, Links, and Buckets are deep-copied, but all Tensors are left as references.
		'''

		raise NotImplementedError