from tensor import Tensor
from treeNetwork import TreeNetwork
from node import Node
from link import Link
from bucket import Bucket
from operator import mul
from copy import deepcopy

class TreeTensor(Tensor):

	def __init__(self):
		'''
		If network is specified, it must be a treeNetwork and must not be connected to any other treeNetworks.
		'''
		self._logScalar = 0.0
		self.network = TreeNetwork()
		self.externalBuckets = []

	def addTensor(self, tensor):
		n = Node(tensor, Buckets=[Bucket() for _ in range(tensor.rank)])
		self.network.addNode(n)
		self.externalBuckets.append(n.buckets)
		if tensor.rank > 3:
			self.network.splitNode(n)

	def __str__(self):
		s = ''
		s = s + 'Tree Tensor with Network:\n'
		s = s + str(self.network)
		return s

	@property
	def shape(self):
		return [b.node.tensor.shape[b.index] for b in self.externalBuckets]

	@property
	def rank(self):
		return len(self.externalBuckets)

	@property
	def size(self):
		size = 1
		for s in self.shape:
			size *= s
		return size

	@property
	def logScalar(self):
		return self._logScalar

	def contract(self, ind, other, otherInd):
		# We copy the two networks first
		t1 = deepcopy(self)
		t2 = deepcopy(other)

		# Link the networks
		links = []
		for i,j in zip(*(ind,otherInd)):
			b1, b2 = t1.externalBuckets[i], t2.externalBuckets[j]
			assert b1 in t1.network.buckets and b1 not in t2.network.buckets
			assert b2 in t2.network.buckets and b2 not in t1.network.buckets
			print(i,j,b1.id,b2.id)
			if b1.linked:
				print(b1.id,b1.link.id,b1.otherBucket.id,b1.otherBucket in t1.network.buckets)
			if b2.linked:
				print(b2.id,b2.link.id,b2.otherBucket.id,b2.otherBucket in t2.network.buckets)
			links.append(Link(b1, b2))

		# Incrementally merge the networks
		while len(links) > 0:
			l = links.pop()
			b1 = l.bucket1
			b2 = l.bucket2

			n1 = b1.node
			n2 = b2.node

			if n1 in t2.network.nodes:
				# Ensure that b1 is in t1 and b2 is in t2, and ditto for n1 and n2
				n2, n1 = n1, n2
				b2, b1 = b1, b2

			# Remove any other links that this node pair handles
			for b in n2.linkedBuckets:
				if b.otherNode in t1.network.nodes and b.link != l:
					links.remove(b.link)

			# Add the links to t2.network to the todo list
			for b in n2.linkedBuckets:
				if b.otherNode in t2.network.nodes:
					links.append(b.link)


			print(n1 in t1.network.nodes, n2 in t2.network.nodes, n1 in t2.network.nodes, n2 in t1.network.nodes)
			print(n1, n2)

			# Move the node over
			t2.network.removeNode(n2)
			t1.network.contractNode(n2)

			for l in links:
				print(l.bucket1.node.id, l.bucket2.node.id, l.id)
				assert l.bucket1 in t1.network.buckets or l.bucket1 in t2.network.buckets
				assert l.bucket2 in t1.network.buckets or l.bucket2 in t2.network.buckets


		#TODO: Handle reindexing of external buckets for t1

		return t1

	def trace(self, ind0, ind1):
		raise NotImplementedError
