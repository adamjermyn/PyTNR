from copy import deepcopy
import numpy as np
import operator

from TNRG.Network.network import Network
from TNRG.Network.node import Node
from TNRG.Network.bucket import Bucket
from TNRG.Network.link import Link
from TNRG.Tensor.arrayTensor import ArrayTensor
from TNRG.Utilities.svd import entropy, splitArray

import sys
sys.setrecursionlimit(10000)

class TreeNetwork(Network):
	'''
	A treeNetwork is a special case of a Network in which the Network being represented
	contains no cycles. This allows matrix elements of a treeNetwork to be efficiently
	evaluated.

	As the only quantities which matter are the matrix elements, the treeNetwork may
	refactor itself through singular value decomposition (SVD) to minimize memory use, and
	so no assumptions should be made about the Nodes in this object, just the external
	Buckets.

	Internally all Nodes of a treeNetwork have Tensors of rank at most 3.
	SVD factoring is used to enforce this.
	'''

	def __init__(self, accuracy=1e-4):
		'''
		treeNetworks require an accuracy argument which determines how accurately (in terms of relative error)
		they promise to represent their matrix elements.
		'''
		super().__init__()

		self.accuracy = accuracy

	def pathBetween(self, node1, node2, calledFrom=None):
		'''
		Returns the unique path between node1 and node2.
		This is done by treating node1 as the root of the binary tree and performing a depth-first search.
		Note that this search only iterates through the internal buckets in the network: it will not consider
		nodes in another network.
		'''
		if node1 == node2: # Found it!
			return [node1]

		if len(self.internalConnected(node1)) == 1 and calledFrom is not None:  # Nothing left to search
			return []

		for c in self.internalConnected(node1): # Search children
			l = node1.findLink(c)
			if c is not calledFrom:
				path = self.pathBetween(c, node2, calledFrom=node1)
				if len(path) > 0: # Means the recursive call found it
					path2 = [node1] + path

					if len(path2) > 0:
						assert node1 in path2
						assert node2 in path2

					return path2

		return []

	def trace(self, b1, b2):
		'''
		Links external buckets b1 and b2 and eliminates any loops which result.
		'''
		assert b1 in self.externalBuckets
		assert b2 in self.externalBuckets
		assert b1 != b2
		n1 = b1.node
		n2 = b2.node

		if n1 == n2:
			# So we're just tracing an arrayTensor.
			n1.tensor = n1.tensor.trace([b1.index], [b2.index])
			n1.buckets.remove(b1)
			n1.buckets.remove(b2)
			self.externalBuckets.remove(b1)
			self.externalBuckets.remove(b2)
		else:
			# We may be introducing a loop
			loop = self.pathBetween(n1, n2)
			if len(loop) > 0:
				if len(loop) == 2:
					# This special case is not possible when contracting in a new node.
					# The easy way to handle it is just to merge the two nodes and then
					# split them if the resulting rank is too high.
					_ = Link(b1, b2)
					n = self.mergeNodes(n1, n2)
					self.splitNode(n)
				else:
					_ = Link(b1, b2)
					self.eliminateLoop(loop)

	def splitNode(self, node, ignore=None):
		'''
		Takes as input a Node and ensures that it is at most rank 3 by factoring rank 3 tensors
		out of it until what remains is rank 3. The factoring is done via a greedy algorithm,
		where the pair of indices with the least correlation with the rest are factored out.
		This is determined by explicitly tracing out all but those indices from the density
		matrix and computing the entropy.

		ignore may be None or a pair of indices.
		In the latter case, the pair of indices will be required to stay together.
		This is enforced by having the pair be the first one factored.
		'''
		nodes = []

		while node.tensor.rank > 3:
			self.removeNode(node)

			array = node.tensor.scaledArray

			s = []

			if ignore is not None:
				p = ignore
				ignore = None
			else:
				p = entropy(array)

			u, v, indices1, indices2 = splitArray(array, p, accuracy=self.accuracy)

			if u.shape[-1] > 1:
				b1 = Bucket()
				b2 = Bucket()
				n1 = Node(ArrayTensor(u, logScalar=node.tensor.logScalar/2), Buckets=[node.buckets[i] for i in indices1] + [b1])
				n2 = Node(ArrayTensor(v, logScalar=node.tensor.logScalar/2), Buckets=[b2] + [node.buckets[i] for i in indices2])
				_ = Link(b1,b2) # This line has to happen before addNode to prevent b1 and b2 from becoming externalBuckets
			else:
				# Cut link
				u = u[...,0]
				v = v[0]
				n1 = Node(ArrayTensor(u, logScalar=node.tensor.logScalar/2), Buckets=[node.buckets[i] for i in indices1])
				n2 = Node(ArrayTensor(v, logScalar=node.tensor.logScalar/2), Buckets=[node.buckets[i] for i in indices2])

			self.addNode(n1)
			self.addNode(n2)
			nodes.append(n1)

			node = n2

		nodes.append(node)

		return nodes

	def eliminateLoop(self, loop): 
		''' 
		Takes as input a list of Nodes which have been linked in a loop. 
		The nodes are assumed to be in linkage order (i.e. loop[i] and loop[i+1] are linked), 
		and the list is assumed to wrap-around (so loop[0] and loop[-1] are linked). 
 
		The loop is assumed to be the only loop in the Network. 
 
		The loop is eliminated by iteratively contracting along the loop and factoring out 
		extra indices as memory requires. This proceeds until the loop has length 3, and then 
		one of the three links is cut via SVD (putting all of that link's entropy in the remaining 
		two links). 
 
		The links are contracted in descending size order. 
		''' 
		for i in range(len(loop)): 
			assert loop[i-1] in loop[i].connectedNodes 
			assert loop[i] in self.nodes 
 
		assert len(loop) >= 3 
 
		print('Loop:',len(loop)) 
		for l in loop: 
			print(l.id,l.tensor.shape,l.tensor.size) 
 
		while len(loop) > 3: 
			print('Loop:',len(loop)) 
			best = [0,0] 
			for i in range(len(loop)): 
				n1 = loop[(i + 1)%len(loop)] 
				n2 = loop[(i + 2)%len(loop)] 
				assert n1 in n2.connectedNodes 
				assert n2 in loop[(i+3)%len(loop)].connectedNodes 
				ind1 = n1.indexConnecting(loop[i]) 
				ind2 = n2.indexConnecting(loop[(i+3)%len(loop)]) 
				b1 = n1.buckets[ind1] 
				b2 = n2.buckets[ind2] 
				if n1.findLink(n2).bucket1.size > best[0]: 
					best[0] = n1.findLink(n2).bucket1.size 
					best[1] = [i, n1, n2, ind1, ind2, b1, b2] 
 
			i, n1, n2, ind1, ind2, b1, b2 = best[1] 
 
			loop = loop[i:] + loop[:i] 
 
			assert loop[0] != loop[1] 
			assert loop[1] != loop[2] 
			assert loop[2] != loop[3] 
			links = n1.linksConnecting(n2) 
			for l in links: 
				assert l != b1.link 
				assert l != b2.link 
				assert l.bucket1 != b1 
				assert l.bucket2 != b1 
				assert l.bucket1 != b2 
				assert l.bucket2 != b2 
 
			print('Loop: Merging',n1.tensor.shape,n2.tensor.shape,ind1,ind2,n1.findLink(n2).bucket1.size) 
			n = self.mergeNodes(n1, n2) 
 
			loop.pop(1) 
 
			if n.tensor.rank > 4 or (n.tensor.size > 1e5 and n.tensor.rank > 3): 
				assert b1 in n.buckets 
				assert b2 in n.buckets 
				assert b1.node is n 
				assert b2.node is n 
				print('Loop: Splitting',n.tensor.shape) 
				nodes = self.splitNode(n, ignore=[n.bucketIndex(b1),n.bucketIndex(b2)]) 
				print('Loop split. Size ratio:',1.0*sum(q.tensor.size for q in nodes)/(n.tensor.size)) 
				for p in nodes: 
					print(p.tensor.shape) 
				n = nodes[0] # The ignored indices always end up in the first node 
 
			loop[1] = n 
 
		# This is necessary because we've been rotating the loop around and so have 
		# no guarantee that the rank conditions have been preserved. 
		for i in range(3): 
			if loop[i].tensor.rank > 3: 
				ind1 = loop[i].indexConnecting(loop[i-1]) 
				ind2 = loop[i].indexConnecting(loop[(i+1)%3]) 
				nodes = self.splitNode(loop[i], ignore=[ind1, ind2]) 
				loop[i] = nodes[0] 
 
		assert loop[0].tensor.rank <= 3 
		assert loop[1].tensor.rank <= 3 
		assert loop[2].tensor.rank <= 3 
 
		n = self.mergeNodes(loop[0], loop[1]) 
		n = self.mergeNodes(n, loop[2]) 
		if n.tensor.rank > 3: 
			self.splitNode(n) 
 
		for n in self.nodes: 
			assert n.tensor.rank <= 3 

 
