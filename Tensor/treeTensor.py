from tensor import Tensor
from arrayTensor import ArrayTensor
from treeNetwork import TreeNetwork
from node import Node
from link import Link
from bucket import Bucket
from operator import mul
from copy import deepcopy
import numpy as np

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
		self.externalBuckets.extend(n.buckets)
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
			links.append(Link(b1, b2))

		for l in links:
			t1.externalBuckets.remove(l.bucket1)
			t2.externalBuckets.remove(l.bucket2)

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

			# Move the node over
			t2.network.removeNode(n2)
			t1.network.contractNode(n2)

			for l in links:
				assert l.bucket1 in t1.network.buckets or l.bucket1 in t2.network.buckets
				assert l.bucket2 in t1.network.buckets or l.bucket2 in t2.network.buckets

			self.optimize()

		t1.externalBuckets = t1.externalBuckets + t2.externalBuckets

		return t1

	def trace(self, ind0, ind1):
		t = deepcopy(self)

		b1 = t.externalBuckets[ind0]
		b2 = t.externalBuckets[ind1]

		t.network.trace(b1, b2)

		t.externalBuckets.remove(b1)
		t.externalBuckets.remove(b2)

	def flatten(self, inds):
		'''
		This method merges the listed external indices using a tree tensor
		by attaching the identity tensor to all of them and to a new
		external bucket. It then returns the new tree tensor.
		'''

		buckets = [self.externalBuckets[i] for i in inds]
		shape = [self.shape[i] for i in inds]

		# Create identity array
		shape.append(np.product(shape))
		iden = np.identity(shape[-1])
		iden = np.reshape(iden, shape)

		# Create Tree Tensor holding the identity
		tens = ArrayTensor(iden)
		tn = TreeTensor()
		tn.addTensor(tens)

		# Contract the identity
		ttens = self.contract(inds, tn, list(range(len(buckets))))

		return ttens

	def optimize(self):
		'''
		Random note:
		Instead of carefully eliminating loops one at a time,
		we could just merge everything wholesale, use networkx
		to identify loops, and then contract them one at a time
		starting with the smallest (this will likely produce
		simplifications by shrinking bigger loops... at a minimum
		it means we're dealing with a smaller network by the time
		we perform the complicated loop elimination operations).

		We'd still need to occasionally split nodes when they get
		too large, but that's something that can be done on the basis
		of size rather than rank. We can then do a pass through to
		enforce the maximum rank condition at the end. 
		'''
		print('Optimizing links...')

		s2 = 0
		for n in self.network.nodes:
			s2 += n.tensor.size

		done = set()
		while len(done.intersection(self.network.nodes)) < len(self.network.nodes):
			n = next(iter(self.network.nodes.difference(done)))
			nc = self.network.internalConnected(n)
			if len(nc) > 0:
				n1 = nc.pop()
				n = self.network.mergeNodes(n, n1)
				nodes = self.network.splitNode(n)
				done.update(nodes)
			else:
				done.add(n)

		print('Optimizing permutations...')

		s = 0
		for n in self.network.nodes:
			s += n.tensor.size

		done = set()
		while len(done.intersection(self.network.nodes)) < len(self.network.nodes):
			n = next(iter(self.network.nodes.difference(done)))
			nc = self.network.internalConnected(n)
			if len(nc) > 1:
				n1 = nc.pop()
				n2 = nc.pop()
				n = self.network.mergeNodes(n, n1)
				n = self.network.mergeNodes(n, n2)
				nodes = self.network.splitNode(n)
				done.update(nodes)
			else:
				done.add(n)

		s1 = 0
		for n in self.network.nodes:
			s1 += n.tensor.size

		for b in self.externalBuckets:
			print('External Bucket:',b.size)
		for b in self.network.internalBuckets:
			print('Internal Bucket:',b.size)

		print('Opt:',s2, s, s1, n.tensor.shape, len(self.network.nodes))
		return s, s1



