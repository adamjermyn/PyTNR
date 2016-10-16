from tensor import Tensor
from treeNetwork import treeNetwork
from node import Node
from link import Link
from bucket import Bucket
from operator import mul
from copy import deepcopy
from factor import iterativeSplit

class treeTensor(Tensor):

	def __init__(self, network=None):
		'''
		If network is specified, it must be a treeNetwork and must not be connected to any other treeNetworks.
		'''
		self._shape = ()
		self._size = 1
		self._rank = 0
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
			self._rank = len(shape)
			self._size = reduce(mul, shape)

	@property
	def shape(self):
		return self._shape

	@property
	def rank(self):
		return self._rank

	@property
	def size(self):
		return self._size

	@property
	def logScalar(self):
		return self._logScalar

	def contract(self, ind, other, otherInd):
		# We copy the two networks first
		net1 = deepcopy(self.network)
		net2 = deepcopy(other.network)

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

			# Check for loops involving n1 and n2
			nodes = []
			for c in n2.connected():
				if c in net1.nodes:
					nodes.append(c)
			if len(nodes) == 2: # Indicates a loop, so we find it
				loop = net1.pathBetween(nodes[0], nodes[1])
			elif len(nodes) > 2:
				raise ValueError('Too many connections for a single node!')

			# Move the Node over
			net2.deregisterNode(n2)
			net1.registerNode(n2)

			# Add new Links to be processed
			for b in n2.buckets:
				if b.link is not None and b.link.otherBucket(b).node in net2:
					links.append(b.link)

			# Eliminate loop
			if len(nodes) == 2 and len(loop) > 0:
				# The second condition allows us to ignore the case where no loop formed despite two nodes connecting.
				# This corresponds to the case where the network contains multiple disjoint trees.
				net1.eliminateLoop(loop + [n2])

		return treeTensor(net1)

def tensorTreeFromArrayTensor(tensor):
	assert hasattr(tensor, 'array')

	network = treeNetwork()

	if len(tensor.shape) <= 3:
		n = Node(tensor, network, Buckets=[Bucket() for _ in tensor.shape])
		return network
	else:
		tree = iterativeSplit(tensor.array)
		nodeTree = []

		todo = [tree]
		while len(todo) > 0:
			t = todo[0]
			for tr in t[1:]:
				todo.append(tr)
			n = Node(t[0], network, Buckets=[Bucket() for _ in t[0].shape])
