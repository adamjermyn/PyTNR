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
		self._logScalar = 0.0
		self.network = network
		self.externalBuckets = []
		for b in self.network.externalBuckets:
			self.externalBuckets.append(b)

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

		# Link the Networks
		links = []
		for i,j in zip(*(ind,otherInd)):
			b1, b2 = t1.externalBuckets[i], t2.externalBuckets[j]
			links.append(Link(b1, b2))

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

			print(n1 in t1.network.nodes, n2 in t2.network.nodes)

			# There are six cases we could be in. We label them by the tuple (T1,T2,E),
			# where T1 is the number of links to t1.network, T2 is the number of links to
			# t2.network, and E is the number of external unlinked buckets on n2.
			# The cases are: (1,0,2), (1,1,1), (1,2,0), (2,0,1), (2,1,0), and (3,0,0).
			# No other cases are allowed because n2 is rank 3 and must have at least
			# one link to t1.network.
			# Fortunately we don't need to consider these cases separately, we just need
			# to handle them by the value of T1.

			nt2 = len(t2.network.internalConnected(n2))
			nt1 = len(n2.linkedBuckets) - nt2
			nE = 3 - nt1 - nt2

			print(nt1,nt2,nE)
			print(t1)
			print(t2)
			print(n1)
			print(n2)

			if nt1 == 1:
				# This case conveniently means that there are no loops introduced by n2,
				# so the process of including it is straightforward.

				# Move the node over
				t2.network.removeNode(n2)
				t1.network.addNode(n2)

				# Add the links to t2.network to the todo list
				for n in n2.connectedNodes:
					if n in t2.network.nodes:
						links.append(n2.findLink(n))
			elif nt1 == 2:
				# This case means that there is a loop introduced by n2,
				# so we first find the loop.
				nodes = []
				for c in n2.connectedNodes:
					if c in t1.network.nodes:
						nodes.append(c)
						if c != n1:
							l = c.findLink(n2)
							links.remove(l)
				nodes = list(set(nodes))
				loop = t1.network.pathBetween(nodes[0], nodes[1])

				# Next we move the node over.
				t2.network.removeNode(n2)
				t1.network.addNode(n2)

				# Now we add links to t2.network to the todo list
				for n in n2.connectedNodes:
					if n in t2.network.nodes:
						links.append(n2.findLink(n))

				# Finally we eliminate the loop
				t1.network.eliminateLoop(loop + [n2])
			elif nt1 == 3:
				# This case means that there are two loops introduced by n2.
				# We get around this by manipulating the externalBucket lists
				# to make one loop appear first, then eliminate it, then
				# allow the second loop to appear.				

				# First we find the nodes involved in the loops
				nodes = []
				for c in n2.connectedNodes:
					if c in t1.network.nodes:
						nodes.append(c)
						if c != n1:
							l = c.findLink(n2)
							links.remove(l)
				nodes = list(set(nodes))
				
				# Now we pick two of them, thereby defining the first loop.
				loop = t1.network.pathBetween(nodes[0], nodes[1])

				# Next we move the node over.
				t2.network.removeNode(n2)
				t1.network.addNode(n2)

				# We leave one bucket behind though, that corresponding to nodes[2].
				link = n2.findLink(nodes[2])
				if link.bucket1.otherBucket in nodes[2].buckets:
					b = link.bucket1
				else:
					b = link.bucket2
				b2 = b.otherBucket
				t1.network.internalBuckets.remove(b)
				t1.network.externalBuckets.add(b)
				t1.network.internalBuckets.remove(b2)
				t1.network.externalBuckets.add(b2)

				# There are no more links from this node.

				# Now we eliminate the first loop
				t1.network.eliminateLoop(loop + [n2])

				# Now we find the second loop
				loop = t1.network.pathBetween(b.node, b2.node)

				# Now we undo our trickery with the buckets
				t1.network.internalBuckets.add(b)
				t1.network.externalBuckets.remove(b)
				t1.network.internalBuckets.add(b2)
				t1.network.externalBuckets.remove(b2)

				# Now we eliminate the second loop
				t1.network.eliminateLoop(loop) 	# Nothing to add here because we used two
												# connected nodes in defining loop this time.

		#TODO: Handle reindexing of external buckets for t1

		return t1

	def trace(self, ind0, ind1):
		raise NotImplementedError
