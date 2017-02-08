from operator import mul
from copy import deepcopy
from collections import defaultdict

import itertools as it
import numpy as np
import operator
import networkx

from TNRG.Tensor.tensor import Tensor
from TNRG.Tensor.arrayTensor import ArrayTensor
from TNRG.Network.treeNetwork import TreeNetwork
from TNRG.Network.node import Node
from TNRG.Network.link import Link
from TNRG.Network.bucket import Bucket
from TNRG.Network.cycleBasis import cycleBasis
from TNRG.Utilities.svd import entropy

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
		s = s + 'Tree Tensor with Shape:' + str(self.shape)+' and Network:\n'
		s = s + str(self.network)
		return s

	@property
	def array(self):
		arr, bdict = self.network.array

		perm = []
		blist = [b.id for b in self.externalBuckets]

		for b in blist:
			perm.append(bdict[b])

		arr = np.transpose(arr, axes=perm)

		assert arr.shape == tuple(self.shape)
		return arr

	@property
	def shape(self):
		return tuple([b.node.tensor.shape[b.index] for b in self.externalBuckets])

	@property
	def rank(self):
		return len(self.externalBuckets)

	@property
	def size(self):
		return np.product(self.shape)

	@property
	def compressedSize(self):
		size = 0
		for n in self.network.nodes:
			size += n.tensor.size
		return size

	def distBetween(self, ind1, ind2):
		n1 = self.externalBuckets[ind1].node
		n2 = self.externalBuckets[ind2].node
		return len(self.network.pathBetween(n1, n2))

	def contract(self, ind, other, otherInd, front=True):
		# This method could be vastly simplified by defining a cycle basis class

		# We copy the two networks first. If the other is an ArrayTensor we cast it to a TreeTensor first.
		t1 = deepcopy(self)
		if hasattr(other, 'network'):
			t2 = deepcopy(other)
		else:
			t2 = TreeTensor(self.accuracy)
			t2.addTensor(other)

		# If front == True then we contract t2 into t1, otherwise we contract t1 into t2.
		# This is so we get the index order correct. Thus
		if not front:
			t1, t2 = t2, t1
			otherInd, ind = ind, otherInd

		# Link the networks
		links = []
		for i,j in zip(*(ind,otherInd)):
			b1, b2 = t1.externalBuckets[i], t2.externalBuckets[j]
			assert b1 in t1.network.buckets and b1 not in t2.network.buckets
			assert b2 in t2.network.buckets and b2 not in t1.network.buckets
			links.append(Link(b1, b2))

		# Determine new external buckets list
		for l in links:
			t1.externalBuckets.remove(l.bucket1)
			t2.externalBuckets.remove(l.bucket2)

		extB = t1.externalBuckets + t2.externalBuckets

		# Merge the networks
		toRemove = set(t2.network.nodes)

		for n in toRemove:
			t2.network.removeNode(n)

		for n in toRemove:
			t1.network.addNode(n)

		# Merge any rank-1 or rank-2 objects
		done = set()
		while len(done.intersection(t1.network.nodes)) < len(t1.network.nodes):
			n = next(iter(t1.network.nodes.difference(done)))
			if n.tensor.rank <= 2:
				nodes = t1.network.internalConnected(n)
				if len(nodes) > 0:
					t1.network.mergeNodes(n, nodes.pop())
				else:
					done.add(n)
			else:
				done.add(n)


		t1.externalBuckets = extB
		assert t1.network.externalBuckets == set(t1.externalBuckets)

		for n in t1.network.nodes:
			assert n.tensor.rank <= 3

		return t1

	def eliminateLoops(self):
		cb = cycleBasis(self.network)

		prevDone = set()
		elim = None

		while len(cb.cycles) > 0:
			print('LOGGING:::::::::::::::',len(cb.cycles),len(cb.smallest()),len(cb.hardest()))
			print(cb)
			# Merge triangles and smaller preferentially because these can be done at no complexity cost
			smallest = cb.smallest()
			if len(smallest) <= 3:
				cb.mergeSmall(smallest)
			else:

				# If there are no triangles or lines we perform a pinch operation
				# on the largest cycle. We prioritize index pairs which are not part
				# of another cycle and we prioritize among these by distance. Among
				# those which are shared with another cycle we prioritize them by distance.
				# Finally if there are no pairs of indices which are in the same cycle or no
				# cycle then we just merge an arbitrarily chosen adjacent pair.

				# In doing this one the hardest cycle at a time we can end with oscillatory behaviour wherein
				# the hardest cycle pinches off and the second hardest cycle increases in size, which is then
				# reversed in the next step. The way we deal with this is to detect when the harest cycle is
				# oscillating without shrinking the network and then eliminate that cycle in its entirety.
				if elim in cb.cycles:
					cycle = elim
				else:
					cycle = cb.smallest()
#					cycle = cb.hardest()
					elim = None

				if (len(cycle), cycle) in prevDone:
					elim = cycle

				# Make sure there are no double links in the cycle
				nodes = cycle.nodes
				i = 0
				skip = False
				while i < len(nodes):
					n = nodes[i]
					i += 1
					if len(n.linkedBuckets) > len(n.connectedNodes) and len(n.connectedNodes) > 1:
						cn = list(n.connectedNodes)
						if len(n.linksConnecting(cn[0])) == 2 and cn[0] in nodes:
							cb.mergeEdge(n.findLink(cn[0]))
						elif len(n.linksConnecting(cn[1])) == 2 and cn[1] in nodes:
							cb.mergeEdge(n.findLink(cn[1]))
						if len(cycle) > 0:
							nodes = cycle.nodes
						else:
							nodes = []
						i = 0
						skip = True
					elif len(n.buckets) == 2:
						cn = list(n.connectedNodes)
						cb.mergeEdge(n.findLink(cn[0])) # Can be improved to make it pick the smaller option						
						if len(cycle) > 0:
							nodes = cycle.nodes
						else:
							nodes = []							
						i = 0
						skip = True

				# If we found double links then we are not necessarily working on the right cycle
				if not skip:
					prevDone.add((len(cycle), cycle))

					# Look for free indices
					freeNodes = cb.freeNodes(cycle)
					if len(freeNodes) > 1:
						pairs = []
						nodes = cycle.nodes

						for n in freeNodes:
							for m in freeNodes:
								if n != m:
									dist = cycle.dist(n,m)
									pairs.append((n,m,dist))
					else:
						pairs = cb.commonNodes(cycle)

					bestPair = min(pairs, key=operator.itemgetter(2))

					edge = cb.makeAdjacent(cycle, bestPair[0], bestPair[1])
					n1 = edge.bucket1.node
					n2 = edge.bucket2.node
					cb.pinch(cycle, n1, n2)



	def trace(self, ind0, ind1):
		'''
		Takes as input:
			ind0	-	A list of indices on one side of their Links.
			ind1	-	A list of indices on the other side of their Links.

		Elements of ind0 and ind1 must correspond, such that the same Link is
		represented by indices at the same location in each list.

		Elements of ind0 should not appear in ind1, and vice-versa.

		Returns a Tensor containing the trace over all of the pairs of indices.
		'''
		arr = self.array

		ind0 = list(ind0)
		ind1 = list(ind1)

		t = deepcopy(self)

		for i in range(len(ind0)):
			b1 = t.externalBuckets[ind0[i]]
			b2 = t.externalBuckets[ind1[i]]

			t.network.trace(b1, b2)

			t.externalBuckets.remove(b1)
			t.externalBuckets.remove(b2)

			for j in range(len(ind0)):
				d0 = 0
				d1 = 0

				if ind0[j] > ind0[i]:
					d0 += 1
				if ind0[j] > ind1[i]:
					d0 += 1
	
				if ind1[j] > ind0[i]:
					d1 += 1
				if ind1[j] > ind1[i]:
					d1 += 1

				ind0[j] -= d0
				ind1[j] -= d1

		return t

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

		shape2 = [self.shape[i] for i in range(self.rank) if i not in inds]
		shape2.append(shape[-1])
		for i in range(len(shape2)):
			assert ttens.shape[i] == shape2[i]

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

			print('Optimizing:',n1.id,n2.id,n1.tensor.shape,n2.tensor.shape)

			t, buckets = self.network.dummyMergeNodes(n1, n2)
			arr = t.array
			if n1.tensor.rank == 3:
				ss = set([0,1])
			elif n2.tensor.rank == 3:
				ss = set([2,3])
			else:
				ss = None

			best = entropy(arr, pref=ss)

			if set(best) != ss and set(best) != set(range(n1.tensor.rank+n2.tensor.rank-2)).difference(ss):
				n = self.network.mergeNodes(n1, n2)
				nodes = self.network.splitNode(n, ignore=best)
				assert len(nodes) == 2
				for b in nodes[0].buckets:
					self.optimized.discard(b)
				for b in nodes[1].buckets:
					self.optimized.discard(b)

				l = nodes[0].findLink(nodes[1])
				self.optimized.add(l.bucket1)
				self.optimized.add(l.bucket2)
				print('Optimizer improved to shapes:',nodes[0].tensor.shape,nodes[1].tensor.shape)
			else:
				self.optimized.add(b1)
				self.optimized.add(b2)

			if verbose >= 1:
				print('Optimization steps left:',-len(self.optimized.intersection(self.network.internalBuckets)) + len(self.network.internalBuckets))

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


