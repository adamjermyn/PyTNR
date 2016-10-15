from tensor import Tensor
from treeNetwork import treeNetwork
from node import Node
from link import Link
from operator import mul

class treeTensor(Tensor):

	def __init__(self, network=None):
		'''
		If network is specified, it must be a treeNetwork and must not be connected to any other treeNetworks.
		'''
		self._shape = ()
		self._size = 1
		self._logScalar = 0.0

		if network is None:
			self.network = treeNetwork()
		else:
			self.network = network

			shape = []
			for b in network.externalBuckets:
				n = b.node
				shape.append(n.tensor.shape[n.bucketIndex(b)])
			self._shape = tuple(shape)
			self._size = reduce(mul, shape)


	@property
	def shape(self):
		return self.shape

	@property
	def size(self):
		return self._size

	@property
	def logScalar(self):
		return self._logScalar

	def eliminateLoop(self, nodes):
		raise NotImplementedError

	def contract(self, ind, other, otherInd):
		# We copy the two networks first
		net1 = self.network.copy()
		net2 = other.network.copy()

		# We then connect the two networks at the relevant places
		links = []
		for i,j in zip(*(ind,otherInd)):
			links.append(Link(net1.externalBuckets[i],net2.externalBuckets[j]))

		# Iterate through all links dividing the networks until none are left.
		while len(links) > 0:
			link = links[0]

			n1 = link.bucket1.node
			n2 = link.bucket2.node

			if n1 not in net1.nodes: # Ensure that n1 is in net1 and that n2 is in net2
				n1, n2 = n2, n1

			# Move the Node over
			net2.deregisterNode(n2)
			net1.registerNode(n2)

			# Add new Links to be processed
			for b in n2.buckets:
				if b.link is not None and b.link.otherBucket(b).node in net2:
					links.append(b.link)

			# Check for loops involving n1 and n2
			nodes = []
			for c in n2.connected():
				if c in net1.nodes:
					nodes.append(c)
			if len(nodes) > 1: # Indicates a loop
				self.eliminateLoop(nodes)

		return treeTensor(net1)