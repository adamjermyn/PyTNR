from tensor import Tensor
from treeNetwork import TreeNetwork
from node import Node
from link import Link
from bucket import Bucket
from operator import mul
from copy import deepcopy

class TreeTensor(Tensor):

	def __init__(self, network):
		'''
		If network is specified, it must be a treeNetwork and must not be connected to any other treeNetworks.
		'''
		self._shape = ()
		self._size = 1
		self._rank = 0
		self._logScalar = 0.0
		self.network = network

		externalBuckets = []
		shape = []
		for n in self.network.nodes:
			for b in n.buckets:
				if b.link is None:
					externalBuckets.append(b)
					shape.append(n.tensor.shape[n.bucketIndex(b)])

		self.externalBuckets = externalBuckets
		self._shape = tuple(shape)
		self._rank = len(shape)
		for s in shape:
			self._size *= s

	def __str__(self):
		s = ''
		s = s + 'Tree Tensor with Network:\n'
		s = s + str(self.network)
		return s

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
		t1 = deepcopy(self)
		t2 = deepcopy(other)

		# We then connect the two networks at the relevant places
		bucketsLeftSelf = []
		bucketsLeftOther = []

		for i,j in zip(*(ind,otherInd)):
			bucketsLeftSelf.append(t1.externalBuckets[i])
			bucketsLeftOther.append(t2.externalBuckets[j])

		# Iterate through all links dividing the networks until none are left.
		while len(bucketsLeftSelf) > 0:
			# Process the next pair of buckets
			b1 = bucketsLeftSelf[0]
			b2 = bucketsLeftOther[0]

			n1 = b1.node
			n2 = b2.node

			# Identify all bucket pairs to be contracted between these nodes.
			# Remove them all from the list of buckets to be processed and link them.

			print(len(bucketsLeftSelf),len(bucketsLeftOther))

			for b1 in n1.buckets:
				if b1 in bucketsLeftSelf:
					ind = bucketsLeftSelf.index(b1)
					b2 = bucketsLeftOther[ind]
					bucketsLeftSelf.remove(b1)
					bucketsLeftOther.remove(b2)
					Link(b1, b2)

			print(len(bucketsLeftSelf),len(bucketsLeftOther))

			if n1 not in t1.network.nodes: # Ensure that n1 is in net1 and that n2 is in net2
				n1, n2 = n2, n1

			print(n1 in t1.network.nodes, n2 in t2.network.nodes, n1 in t2.network.nodes, n2 in t1.network.nodes)
			net1 = t1.network
			net2 = t2.network

			# Check for loops involving n1 and n2
			nodes = []
			for c in n2.connected():
				if c in net1.nodes:
					nodes.append(c)
			nodes = list(set(nodes))
			if len(nodes) == 2: # Indicates a loop, so we find it
				loop = net1.pathBetween(nodes[0], nodes[1])
			elif len(nodes) > 2:
				raise ValueError('Too many connections for a single node!')

			# Move the Node over
			net2.removeNode(n2)
			net1.addNode(n2)

			print(net1)
			print(net2)
			print(n1 in t1.network.nodes, n2 in t2.network.nodes, n1 in t2.network.nodes, n2 in t1.network.nodes)
			print('---')

			for i in range(len(bucketsLeftSelf)):
				assert bucketsLeftSelf[i].node in net1.nodes
				assert bucketsLeftOther[i].node in net2.nodes

			# Add new Links to be processed and remove now-redundant ones
			for b in n2.buckets:
				if b.link is not None and b.otherBucket.node in net2.nodes:
					bucketsLeftSelf.append(b)
					bucketsLeftOther.append(b.otherBucket)
					# Need to erase any existing link to avoid prematurely coupling the rest of net2
					b.otherBucket.link = None
					b.link = None

			for i in range(len(bucketsLeftSelf)):
				assert bucketsLeftSelf[i].node in net1.nodes
				assert bucketsLeftOther[i].node in net2.nodes

			# Eliminate loop
			if len(nodes) == 2 and len(loop) > 0:
				# The second condition allows us to ignore the case where no loop formed despite two nodes connecting.
				# This corresponds to the case where the network contains multiple disjoint trees.
				net1.eliminateLoop(loop + [n2])



		return TreeTensor(net1)

	def trace(self, ind0, ind1):
		raise NotImplementedError
