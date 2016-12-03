from tensor import Tensor
from arrayTensor import ArrayTensor
from treeNetwork import TreeNetwork
from node import Node
from link import Link
from bucket import Bucket
from operator import mul
from copy import deepcopy
import numpy as np
import operator


class TreeTensor(Tensor):

	def __init__(self, accuracy):
		self.accuracy = accuracy
		self.network = TreeNetwork(accuracy=accuracy)
		self.externalBuckets = []
		self.optimized = set()

	def addTensor(self, tensor):
		n = Node(tensor, Buckets=[Bucket() for _ in range(tensor.rank)])
		self.network.addNode(n)
		self.externalBuckets.extend(n.buckets)
		if tensor.rank > 3:
			self.network.splitNode(n)
		return n

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
		size = 0
		for n in self.network.nodes:
			size += n.tensor.size
		return size

	def contract(self, ind, other, otherInd):
		# We copy the two networks first. If the other is an ArrayTensor we cast it to a TreeTensor first.
		t1 = deepcopy(self)
		if hasattr(other, 'network'):
			t2 = deepcopy(other)
		else:
			t2 = TreeTensor(self.accuracy)
			t2.addTensor(other)

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
			# We perform the mergers which can be done on their own first,
			# then prioritize based on the width of the induced loop (smallest first).
			plist = []
			for l in links:
				b1 = l.bucket1
				b2 = l.bucket2
				n1 = b1.node
				n2 = b2.node
				if n1 not in t1.network.nodes:
					n1, n2 = n2, n1
					b1, b2 = b2, b1
				connected = []
				for c in n2.connectedNodes:
					if c in t1.network.nodes:
						connected.append(c)
				if len(connected) == 1:
					plist.append([0,l])
				elif len(connected) == 2:
					plist.append([len(t1.network.pathBetween(connected[0], connected[1])), l])
				else:
					plist.append([100000, l])

			m, l = min(plist, key=operator.itemgetter(0))

			links.remove(l)

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
		tn = TreeTensor(self.accuracy)
		tn.addTensor(tens)

		# Contract the identity
		ttens = self.contract(inds, tn, list(range(len(buckets))))

		return ttens

	def getIndexFactor(self, ind):
		return self.externalBuckets[ind].node.tensor.scaledArray, self.externalBuckets[ind].index

	def setIndexFactor(self, ind, arr):
		tt = deepcopy(self)
		tt.externalBuckets[ind].node.tensor = ArrayTensor(arr, logScalar=tt.externalBuckets[ind].node.tensor.logScalar)
		return tt

	def optimize(self, verbose=0):
		'''
		Optimizes the tensor network to minimize memory usage.
		The parameter verbose controls how much output to print:
			0 - None
			1 - Running status
		'''
		if verbose >= 1:
			print('Starting optimizer.')
			print('Optimizing tensor with shape',self.shape)
			s2 = 0
			for n in self.network.nodes:
				s2 += n.tensor.size

		if verbose >= 1:
			print('Contracting Rank-2 Tensors.')
		done = set()
		while len(done.intersection(self.network.nodes)) < len(self.network.nodes):
			n = next(iter(self.network.nodes.difference(done)))
			if n.tensor.rank == 2:
				nodes = self.network.internalConnected(n)
				if len(nodes) > 0:
					self.network.mergeNodes(n, nodes.pop())
				else:
					done.add(n)
			else:
				done.add(n)

		if verbose >= 1:
			print('Optimizing links.')

		while len(self.optimized.intersection(self.network.internalBuckets)) < len(self.network.internalBuckets):
			b1 = next(iter(self.network.internalBuckets.difference(self.optimized)))
			b2 = b1.otherBucket
			n1 = b1.node
			n2 = b2.node

			sh1 = n1.tensor.shape
			sh2 = n2.tensor.shape
			s = n1.tensor.size + n2.tensor.size
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
						self.optimized.discard(b)
					for b in n2.buckets:
						self.optimized.discard(b)

				self.optimized.add(l.bucket1)
				self.optimized.add(l.bucket2)

			s1 = 0
			s1sh = []
			for nnnn in nodes:
				s1 += nnnn.tensor.size
				s1sh.append(nnnn.tensor.shape)

			if verbose >= 1:
				print('Optimization steps left:',-len(self.optimized.intersection(self.network.internalBuckets)) + len(self.network.internalBuckets))
				print('Tensor changed from',s,sh1,sh2,'to',s1,*s1sh,'\n')


		if verbose >= 1:
			print('Optimized network:')
			s1 = 0
			for n in self.network.nodes:
				print(n)
				s1 += n.tensor.size
			print('Shape: ',self.shape)
			print('Number of internal nodes:',len(self.network.nodes))
			print('Reduced size from',s2,'to',s1)
			print('Optimization done.\n')


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
