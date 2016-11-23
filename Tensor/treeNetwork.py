from network import Network
from node import Node
from bucket import Bucket
from link import Link
from arrayTensor import ArrayTensor
from utils import entropy, splitArray
from itertools import combinations
from copy import deepcopy
import numpy as np

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

	def array(self):
		'''
		Contracts the tree down to an array object.
		Indices are ordered according to the external buckets listed in buckets.
		'''
		net = deepcopy(self)
		while len(net.nodes) > 1:
			n = net.nodes.pop()
			net.nodes.add(n)
			c = net.internalConnected(n)
			c = c.pop()
			net.mergeNodes(n,c)

		n = net.nodes.pop()
		arr = n.tensor.array
		return arr

	def pathBetween(self, node1, node2, calledFrom=None):
		'''
		Returns the unique path between node1 and node2.
		This is done by treating node1 as the root of the binary tree and performing a depth-first search.
		Note that this search only iterates through the internal buckets in the network: it will not consider
		nodes in another network.
		'''
		if node1 == node2: # Found it!
			return [node1]

		if len(self.internalConnected(node1)) == 1 and calledFrom is not None:	# Nothing left to search
			return []

		for c in self.internalConnected(node1): # Search children
			l = node1.findLink(c)
			if c is not calledFrom:
				path = self.pathBetween(c, node2, calledFrom=node1)
				if len(path) > 0: # Means the recursive call found it
					path2 = [node1] + path
					return path2

		return []

	def contractNode(self, n):
		'''
		This method adds the node n to this network.
		This node must be at most of rank 3.
		The node may be linked already to members of this network.
		This method handles the logic of removing any loops which arise in the process.

		This method is distinct from addNode in that it does not simply append the node
		to the network. In that sense this method is more specialised and obeys more
		stringent conditions.
		'''
		assert n.tensor.rank <= 3

		connected = []
		for c in n.connectedNodes:
			if c in self.nodes:
				connected.append(c)

		# Because of the assertion, len(connected) <= 3
		# If len(connected) <= 1 there's nothing tricky for us
		# to do, but if len(connected) > 1 we have to
		# eliminate loops and such.
		if len(connected) == 1:
			self.addNode(n)
		elif len(connected) == 2:
			n1 = connected[0]
			n2 = connected[1]
			if n1 == n2:
				self.addNode(n)
				n = self.mergeNodes(n, n1)
			else:
				# Means there's a loop
				loop = self.pathBetween(n1, n2)
				if len(loop) > 0:
					self.addNode(n)
					self.eliminateLoop(loop + [n])
				else:
					self.addNode(n)
		elif len(connected) == 3:
			'''
			This case is somewhat complicated to handle, so we're going to do it
			in a roundabout way. First we insert a rank-2 identity tensor between
			n and one of the nodes it connects to. Then, we contract n (which sends
			it to the len(connected)==2 case), and finally we contract the identity.
			'''

			# Build the identity and move over bucket, linking it to this network
			b1 = n.buckets[0]
			b2 = Bucket()
			s = b1.size
			identity = Node(ArrayTensor(np.identity(s)), Buckets=[b1, b2])

			# Link the identity to n
			b3 = Bucket()
			n.buckets[0] = b3
			b3.node = n
			_ = Link(b2, b3)

			# Contract n
			self.contractNode(n)
			# Contract the identity
			self.contractNode(identity)


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
					self.eliminateLoop(loop + [n1])
					self.externalBuckets.remove(b1)
					self.externalBuckets.remove(b2)

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
			pairs = list(combinations(range(len(array.shape)), 2))

			if ignore is not None:
				p = ignore
				ignore = None
			else:
				for p in pairs:
					s.append(entropy(array, p))
				ind = s.index(min(s))
				p = pairs[ind]

			u, v, indices1, indices2 = splitArray(array, p, accuracy=self.accuracy)

			b1 = Bucket()
			b2 = Bucket()
			n1 = Node(ArrayTensor(u, logScalar=node.tensor.logScalar/2), Buckets=[node.buckets[i] for i in indices1] + [b1])
			n2 = Node(ArrayTensor(v, logScalar=node.tensor.logScalar/2), Buckets=[b2] + [node.buckets[i] for i in indices2])
			_ = Link(b1,b2) # This line has to happen before addNode to prevent b1 and b2 from becoming externalBuckets

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
		'''
		for i in range(len(loop)):
			assert loop[i-1] in loop[i].connectedNodes

		assert len(loop) >= 3

		while len(loop) > 3:
			n1 = loop[1]
			n2 = loop[2]
			ind1 = n1.indexConnecting(loop[0])
			ind2 = n2.indexConnecting(loop[3])
			b1 = n1.buckets[ind1]
			b2 = n2.buckets[ind2]

			assert loop[0] != loop[1]
			assert loop[1] != loop[2]
			assert loop[2] != loop[3]
			links = n1.linksConnecting(n2)
			for l in links:
				assert l.bucket1 != b1
				assert l.bucket2 != b1
				assert l.bucket1 != b2
				assert l.bucket2 != b2

			l = n1.findLink(n2)
			n = self.mergeNodes(n1, n2)

			loop.pop(1)

			if n.tensor.rank > 3:
				assert b1 in n.buckets
				assert b2 in n.buckets
				assert b1.node is n
				assert b2.node is n
				nodes = self.splitNode(n, ignore=[n.bucketIndex(b1),n.bucketIndex(b2)])
				n = nodes[0] # The ignored indices always end up in the first node

			loop[1] = n

		if loop[1].tensor.rank > 3:
			ind1 = n1.indexConnecting(loop[0])
			ind2 = n2.indexConnecting(loop[2])
			b1 = n1.buckets[ind1]
			b2 = n2.buckets[ind2]
			nodes = self.splitNode(n, ignore=[n.bucketIndex(b1),n.bucketIndex(b2)])
			n = nodes[0]
			loop[1] = n

		assert len(loop) == 3

		n1 = loop[0]
		n2 = loop[1]
		n3 = loop[2]
		n = self.mergeNodes(n1, n2)
		n = self.mergeNodes(n, n3)

		if n.tensor.rank > 3:
			self.splitNode(n)



