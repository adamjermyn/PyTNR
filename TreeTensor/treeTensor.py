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
from TNRG.Network.traceMin import traceMin
from TNRG.Utilities.svd import entropy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

counter0 = 0

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

	def distBetweenBuckets(self, b1, b2):
		n1 = b1.node
		n2 = b2.node
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

		assert t1.rank == self.rank + other.rank - 2*len(ind)

		return t1

	def eliminateLoops(self, otherNodes):
		global counter0
		tm = traceMin(self.network, otherNodes)

#		counter = 0
#		pos = None

		while len(networkx.cycles.cycle_basis(self.network.toGraph())) > 0:
#			g = self.network.toGraph()
#			labels = networkx.get_edge_attributes(g, 'weight')
#			for l in labels.keys():
#				labels[l] = round(labels[l], 0)
#			reusePos = {}
#			if pos is not None:
#				for n in g.nodes():
#					if n in pos:
#						reusePos[n] = pos[n]
#				pos=networkx.fruchterman_reingold_layout(g, pos=reusePos, fixed=reusePos.keys())
#			else:
#				pos=networkx.fruchterman_reingold_layout(g)
#			plt.figure(figsize=(11,11))
#			weights = [g.edge[i][j]['weight']**2/5 for (i,j) in g.edges_iter()]
#			networkx.draw(g, pos, width=weights, edge_color=[cm.jet(w/max(weights)) for w in weights])
#			networkx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
#			plt.savefig('PNG/'+str(counter0) + '_' + str(counter) + '.png')
#			plt.clf()
#			counter += 1

			print('LOGGING:::::::::::::::',tm.util, len(networkx.cycles.cycle_basis(self.network.toGraph())))

			merged = tm.mergeSmall()

			if not merged:
				best = tm.bestSwap()
				print(best[0])
				tm.swap(best[1],best[2],best[3])

#			if False:
#				if len(networkx.cycles.cycle_basis(self.network.toGraph())) == 0 and len(self.network.toGraph().nodes()) > 0:
#					g = self.network.toGraph()
#					labels = networkx.get_edge_attributes(g, 'weight')
#					for l in labels.keys():
#						labels[l] = round(labels[l], 0)
#					reusePos = {}
#					if pos is not None:
#						for n in g.nodes():
#							if n in pos:
#								reusePos[n] = pos[n]
#						pos=networkx.fruchterman_reingold_layout(g, pos=reusePos, fixed=reusePos.keys())
#					else:
#						pos=networkx.fruchterman_reingold_layout(g)
#					plt.figure(figsize=(11,11))
#					weights = [g.edge[i][j]['weight']**2/5 for (i,j) in g.edges_iter()]
#					networkx.draw(g, pos, width=weights, edge_color=[cm.jet(w/max(weights)) for w in weights])
#					networkx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
#					plt.savefig('PNG/'+str(counter0) + '_' + str(counter) + '.png')
#					plt.clf()

			print(self.network)


		counter0 += 1
		assert len(networkx.cycles.cycle_basis(self.network.toGraph())) == 0
#		if self.rank > 2:
#			expRank = sum([n.tensor.rank for n in self.network.nodes]) - 2*len(self.network.nodes) + 2
#			print(expRank)
#			print(self.rank)
#			print(self.network)
#			assert self.rank == expRank

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
			print('Contracting Double Links.')
		done = set()
		while len(done.intersection(self.network.nodes)) < len(self.network.nodes):
			n = next(iter(self.network.nodes.difference(done)))
			nodes = self.network.internalConnected(n)
			merged = False
			for n2 in nodes:
				if len(n.findLinks(n2)) > 1:
					self.network.mergeNodes(n, n2)
					merged = True
			if not merged:
				done.add(n)

#		return

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

			print('Entropy...')
			best = entropy(arr, pref=ss)
			print('Done.')

			if set(best) != ss and set(best) != set(range(n1.tensor.rank+n2.tensor.rank-2)).difference(ss):
				n = self.network.mergeNodes(n1, n2)
				nodes = self.network.splitNode(n, ignore=best)
				print(len(nodes), n.tensor.rank)
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


