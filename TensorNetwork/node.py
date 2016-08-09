from link import Link
from bucket import Bucket
from tensor import Tensor

class Node:
	'''
	A Node object stores the Buckets corresponding to a given Tensor.
	It also provides helper methods allowing for easy access to neighbors.

	Node objects have the following functions which do not modify them:

	id 				-	Returns the id number of the Node. These numbers are unique within a network.
	children		-	Returns the children Nodes (if any) which merged to form this Node.
	parent			-	Returns the Node (if any) which this merges to form.
	network 		-	Returns the Network this Node belongs to.
	connected		-	Returns the Nodes this one is connected to.
	connectedHigh	-	Returns the Nodes this one is connected to, giving the highest-level list possible.
						That is, the list such that no element of the set has a parent which is connected
						to this Node, and such that all Nodes connected to this one are children to some
						degree of an element in the list.
	tensor 			-	Returns the Tensor underlying this Node.
	bucket 			-	Returns the Bucket at the given index.
	buckets 		-	Returns all Buckets.
	bucketIndex 	-	Returns the index of the specified bucket.

	There are functions which modify Nodes by linking them or by setting heirarchy attributes:

	addLink			-	Takes as input another Node as well as the index of the Bucket on this Node
						and the index of the Bucket on the other Node. Links them.

	setParent		-	Sets the parent of this Node to the reference given.

	There are additional functions which create modified copies of Nodes, listed below.
	These may be called only if the node is parentless.

	trace			-	Searches for indices which are linked to one another and produces a new Node
						with them traced out. The new Node is then the parent of this Node.
	merge 			-	Takes as input another Node and merges this Node with it. The net result is that
						a new node is added to the network. This Node and the other are left intact, with
						all of their Links preserved. The rest of the network's Nodes will have Links both
						to these and to the new Node, with Links to the new Node appearing later in their
						Link lists.

	Finally, nodes may be deleted:

	delete			-	Delete the Node and all associated Links. Recursively deletes all parents.

	TODO: 	Add method for copying a node to a higher level (so it can be modified in some fashion).
			This method needs to preserve compressed status for links. This does mean that methods which
			violate compressed status (merge, mergeLinks) need to have a modifier method they can
			use to do so.

	TODO:	Add method mergeLinks, which copies two tensors to a higher level and merges the multiple
			links between them.

	'''
	def __init__(self, tens, network, children=[]):
		self.__tensor = tens
		self.__network = network
		self.__id = self.__network.nextID()
		self.__parent = None
		self.__children = children
		self.__network.registerNode(self)
		for c in children:
			c.setParent(self)
		self.__buckets = [Bucket(self,i,self.__network) for i in range(len(self.__tensor.shape()))]

	def id(self):
		return self.__id

	def children(self):
		return self.__children

	def parent(self):
		return self.__parent

	def setParent(self, parent):
		self.__parent = parent

	def network(self):
		return self.__network

	def connected(self):
		c = []
		for b in self.__buckets:
			if b.linked():
				for i in range(len(b.numLinks())):
					c.append(b.otherNode(i))
		return c

	def connectedHigh(self):
		c = []
		for b in self.__buckets:
			if b.linked():
				c.append(b.otherNode(b.numLinks()-1))
		return c

	def tensor(self):
		return self.__tensor

	def bucketIndex(self, b):
		return self.__buckets.index(b)

	def bucket(self, i):
		return self.__buckets[i]

	def buckets(self):
		return self.__buckets

	def delete(self):
		if self.__parent is not None:
			self.__parent.delete()

		for b in self.__buckets:
			for l in b.links():
				l.delete()

		for c in self.children():
			c.parent = None

		self.__network.deregisterNode(self)

		del self.__buckets
		del self

	def addLink(self, other, selfBucketIndex, otherBucketIndex, compressed=False):
		selfBucket = self.bucket(selfBucketIndex)
		otherBucket = other.bucket(otherBucketIndex)

		l = Link(selfBucket,otherBucket,self.__network,compressed=compressed)

		selfBucket.addLink(l)
		otherBucket.addLink(l)

		return l

	def trace(self):
		for b in self.__buckets:
			if b.linked():
				otherBucket = b.otherBucket(-1)
				otherNode = otherBucket.node()
				if otherNode == self:
					ind0 = self.bucketIndex(b)
					ind1 = self.bucketIndex(otherBucket)
					newT = self.__tensor.trace(ind0,ind1)
					n = Node(newT,self.__network,children=[self])
					counter = 0
					for bb in self.__buckets:
						if bb.linked() and bb != b and bb != otherBucket:
							n.addLink(bb.otherNode(-1),counter,bb.otherNode(-1).bucketIndex(bb.otherBucket(-1)))
							counter += 1
					n.trace() # Keep going until there are no more repeated indices to trace.
					return

	def merge(self, other):
		print self.id(),other.id()
		c =self.connectedHigh()
		if other not in c:
			raise ValueError # Only allow mergers between highest-level objects (so each Node has at most one parent).

		# Find all links between self and other
		links = []
		for i,b in enumerate(self.__buckets):
			if b.linked():
				if b.otherNode(-1) == other:
					links.append((i,other.bucketIndex(b.otherBucket(-1))))

		links = zip(*links)

		# Contract along common links
		t = self.__tensor.contract(links[0],other.tensor(),links[1])

		# Build new Node
		n = Node(t,self.__network,children=[self,other])

		# Link new Node
		counter = 0

		for i in range(len(self.tensor().shape())):
			b = self.bucket(i)
			if i not in links[0]:
				if b.linked():
					n.addLink(b.otherNode(-1),counter,b.otherNode(-1).bucketIndex(b.otherBucket(-1)))
				counter += 1
		for i in range(len(other.tensor().shape())):
			b = other.bucket(i)
			if i not in links[1]:
				if b.linked():
					n.addLink(b.otherNode(-1),counter,b.otherNode(-1).bucketIndex(b.otherBucket(-1)))
				counter += 1
	
