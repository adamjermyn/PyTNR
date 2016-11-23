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

	def getIndexFactor(self, ind):
		return self.externalBuckets[ind].node.tensor.array, self.externalBuckets[ind].index

	def setIndexFactor(self, ind, arr):
		tt = deepcopy(self)
		tt.externalBuckets[ind].node.tensor = ArrayTensor(arr)
		return tt

	def optimize(self):
		print('Optimizing links...')

		s2 = 0
		for n in self.network.nodes:
			s2 += n.tensor.size

		done = set()
		while len(done.intersection(self.network.internalBuckets)) < len(self.network.internalBuckets):
			b1 = next(iter(self.network.internalBuckets))
			b2 = b1.otherBucket
			n1 = b1.node
			n2 = b2.node


			n = self.network.mergeNodes(n1, n2)
			nodes = self.network.splitNode(n)
			if len(nodes) > 1:
				l = nodes[0].findLink(nodes[1])

				newBuckets1 = set(nodes[0].buckets)
				newBuckets1.discard(l.bucket1)
				newBuckets1.discard(l.bucket2)

				newBuckets2 = set(nodes[1].buckets)
				newBuckets2.discard(l.bucket1)
				newBuckets2.discard(l.bucket2)

				oldBuckets1 = set(n1.buckets)
				oldBuckets1.discard(b1)
				oldBuckets1.discard(b2)

				if newBuckets1 != oldBuckets1 and newBuckets2 != oldBuckets1:
					# Means we've done something so all the other buckets on these
					# nodes need to be reexamined.
					for b in n1.buckets:
						done.discard(b)
					for b in n2.buckets:
						done.discard(b)

				done.add(l.bucket1)
				done.add(l.bucket2)
			# It's pretty clear that it's getting stuck in a loop of moving
			# legs around, so that's probably something to fix...
			print(-len(done.intersection(self.network.internalBuckets)) + len(self.network.internalBuckets))

		s=0
		s1 = 0
		for n in self.network.nodes:
			s1 += n.tensor.size

		for b in self.network.internalBuckets:
			print('Internal Bucket:',b.size)

		print('Opt:',s2, s, s1, self.shape, len(self.network.nodes))
		return s, s1


'''

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

'''
