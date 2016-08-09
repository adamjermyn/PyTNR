import numpy as np

class Link:
	'''
	A Link is a means of indicating an intent to contract two tensors.
	This object has two Buckets, one for each Node being connected.
	In addition, it has a method for computing the von Neumann entropy of the Link.
	In cases where this computation is intractable due to memory requirements, a heuristic
	is used.

	Links have the following functions:

	bucket1			-	Returns the first Bucket this link connects to.
	bucket2			-	Returns the second Bucket this link connects to.
	mergeEentropy	-	Returns the expected change in the entropy of the network
						were the link to be contracted. Heuristics are used on the
						assumption that links are being compressed regularly.
	compress		-	Compresses the Link uses Singular Value Decomposition. This produces a new Node on
						either side of the Link, one level up the heirarchy. This is done instead of modifying
						the original Tensors so that subsequent optimization may be done.
	delete			-	Removes this link from both 

	Links are instantiated with the buckets they connect, and are added to the end of the Link
	lists of their buckets. They are also added to the link registry of their TensorNetwork.
	'''

	def __init__(self, b1, b2, network):
		self.__b1 = b1
		self.__b2 = b2
		self.__network = network
		self.__network.registerLink(self)

	def bucket1(self):
		return self.__b1

	def bucket2(self):
		return self.__b2

	def mergeEntropy(self, reduction=0.75):
		# Entropy is computed in base e.
		# As a heuristic we assume that merging a bond of
		# size S with a bond of size S' produces a bond of
		# size reduction*S*S'.

		entF = 8*np.log(2)		# Entropy in a 64-bit float

		n1 = self.__b1.node()
		n2 = self.__b2.node()

		t1 = n1.tensor()
		t2 = n2.tensor()

		arr1 = t1.array()
		arr2 = t2.array()

		length = n1.tensor().shape()[n1.bucketIndex(self.__b1)]

		if self.__b1.node() == self.__b2.node():
			raise ValueError		# You should always trace out self-loops before examining entropy.

		s1 = arr1.size*entF
		s2 = arr2.size*entF

		sN = s1*s2/(entF*length**2)	# Estimate based on no merging of links

		commonNodes = set(n1.connectedHigh()).intersection(set(n2.connectedHigh()))

		commonNodes = list(commonNodes)

		sN *= reduction**len(commonNodes)

		dS = sN - s1 - s2

		return dS

	def compress(self, tolerance):
		raise NotImplementedError

	def delete(self):
		self.__network.deregisterLink(self)
		self.__b1.removeLink(self)
		self.__b2.removeLink(self)

class Bucket:
	'''
	A Bucket is a means of externally referencing an index of a tensor which handles the way in which
	the tensor is linked to other tensors.

	Each Bucket references exactly one index, but may contain multiple Links between that index and others.
	This allows a given tensor to be part of a heirarchical network, wherein nodes may be merged while
	retaining information about the unmerged structure. To accomodate this, each Bucket contains a list
	of Links. When two nodes are merged, the old links remain, while a new Link to the merged object
	is added to the end of the Link list.

	Buckets have the following functions:

	node 		-	Returns the Node this Bucket belongs to.
	index 		-	Returns the index of the Node's Tensor this Bucket refers to.
	network 	-	Returns the TensorNetwork this Bucket belongs to.
	numLinks	-	Returns the number of Links this Bucket has.
	links 		-	Returns all Links this Bucket has.
	link 		-	Takes as input an integer specifying the index of the Link of interest and returns
					that link.
	linked		-	Returns True if the Bucket is linked. Returns False otherwise.
	otherBucket	-	Takes as input an integer specifying the index of the Link of interest and returns
					the Bucket on the other side of that Link.
	otherNode	-	Takes as input an integer specifying the index of the Link of interest and returns
					the Node on the other side of that Link.
	addLink		-	Takes as input a Link and appends it to the end of the Link list.
	removeLink	-	Removes a Link from the Link list. Raises a ValueError if the Link is not present.
	'''

	def __init__(self, node, index, network):
		self.__node = node
		self.__index = index
		self.__network = network

		self.__links = None

	def node(self):
		return self.node

	def index(self):
		return self.index

	def network(self):
		return self.network

	def numLinks(self):
		return len(self.__links)

	def links(self):
		return self.__links

	def link(self, index):
		return self.__links[index]

	def linked(self):
		return (len(self.__links) > 0)

	def otherBucket(self, index):
		b = self.__links[index].bucket1
		if b == self:
			b = self.__links[index].bucket2
		return b

	def otherNode(self, index):
		return self.otherBucket(index).node()

	def addLink(self, link):
		self.__links.append(link)

	def removeLink(self, link):
		if link in self.__links:
			self.__links.remove(link)
		else:
			raise ValueError
