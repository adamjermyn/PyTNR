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

	def delete(self):
		self.__network.deregisterLink(self)
		self.__b1.removeLink(self)
		self.__b2.removeLink(self)
