import numpy as np
from tensor import Tensor

class Link:
	'''
	A Link is a means of indicating an intent to contract two tensors.
	This object has two Buckets, one for each Node being connected.
	In addition, it has a method for computing the von Neumann entropy of the Link.
	In cases where this computation is intractable due to memory requirements, a heuristic
	is used.

	Links have the following functions:

	bucket1			-	Returns the first Bucket this Link connects to.
	bucket2			-	Returns the second Bucket this Link connects to.
	network 		-	Returns the Network this link belongs to.
	parent 			-	Returns the parent of this Link.
	children 		-	Returns the children of this Link.
	setParent		-	Takes as input a Link and sets it as the parent of this Link.
	otherBucket		-	Takes as input a Bucket. Raises a ValueError if it is not one of the
						Buckets associated with this Link. If it is one of them, returns the other.
	mergeEntropy	-	Returns the expected change in the entropy of the network
						were the link to be contracted. Heuristics are used on the
						assumption that links are being compressed regularly. Assumes compression
						would be performed on top-Level nodes.
	updateMergeEntropy	-	Updates the stored merge entropy. Should only be called from the update() method.
	update 			-	Called whenever a Bucket this Link points to gains or loses a Node. Updates merged
						entropy and corrects/registers the top/not top status of the Link.
	compressed 		-	Returns True if this Link was the result of a compression operation, False otherwise.
	setCompressed	-	Sets the Link compress variable to True.
	delete			-	Removes this link from both 

	Links are instantiated with the buckets they connect, and are added to the end of the Link
	lists of their buckets. They are also added to the link registry of their TensorNetwork.
	'''

	def __init__(self, b1, b2, network, compressed=False, optimized=False, reduction=0.75, children=None):
		if children is None:
			children = []
		self.__b1 = b1
		self.__b2 = b2
		self.__compressed = compressed
		self.__optimized = optimized
		self.__network = network
		self.__reduction = reduction
		self.__mergeEntropy = None
		self.__parent = None
		self.__children = children
		for c in self.__children:
			c.setParent(self)
			self.__network.deregisterLinkTop(c)
		self.__network.registerLink(self)

	def bucket1(self):
		return self.__b1

	def bucket2(self):
		return self.__b2

	def network(self):
		return self.__network

	def parent(self):
		return self.__parent

	def setParent(self, parent):
		self.__parent = parent

	def children(self):
		return self.__children

	def otherBucket(self, bucket):
		if bucket == self.__b1:
			return self.__b2
		elif bucket == self.__b2:
			return self.__b1
		else:
			raise ValueError

	def compressed(self):
		return self.__compressed

	def setCompressed(self):
		self.__compressed = True

	def optimized(self):
		return self.__optimized

	def setOptimized(self):
		self.__optimized = True

	def mergeEntropy(self):
		if self.__mergeEntropy is None:
			self.updateMergeEntropy()
		return self.__mergeEntropy

	def updateMergeEntropy(self):
		# Entropy is computed in base e.
		# As a heuristic we assume that merging a bond of
		# size S with a bond of size S' produces a bond of
		# size reduction*S*S'.

		n1 = self.__b1.topNode()
		n2 = self.__b2.topNode()

		t1 = n1.tensor()
		t2 = n2.tensor()

		length = t1.shape()[n1.bucketIndex(self.__b1)]

		if n1 == n2:
			raise ValueError		# You should always trace out self-loops before examining entropy.

		s1 = t1.size()
		s2 = t2.size()

		sN = s1*s2/length**2	# Estimate based on no merging of Links


		# Correction based on number of merging Links
		commonNodes = set(n1.connectedHigh()).intersection(set(n2.connectedHigh()))
		sN *= self.__reduction**len(commonNodes)

		dS = sN - s1 - s2

		self.__mergeEntropy = dS

	def update(self):
		self.updateMergeEntropy()
		if self.__parent is None:
			# I have a feeling there's a more elegant way to handle this, but I'm not sure what it is.
			self.__network.updateSortedLinkList(self)

	def delete(self):
		# This method assumes you want to delete the nodes on either side!
		assert len(self.__children) > 0

		self.__network.deregisterLink(self)
		self.__b1.removeLink()
		self.__b2.removeLink()

		# This means that if we remove the references to these buckets in the Nodes then
		# the buckets will be deleted (there will be no remaining references to them
		# assuming they are in fact only referenced by a single Node each).
		self.__b1 = None
		self.__b2 = None

		for c in self.__children:
			c.setParent(None)
			self.__network.registerLinkTop(c)
