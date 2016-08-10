from link import Link
from bucket import Bucket
from tensor import Tensor
from mergeLinks import mergeLinks

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
	linksConnecting	-	Returns the Links connecting this Node to another (given as input).
	tensor 			-	Returns the Tensor underlying this Node.
	bucket 			-	Returns the Bucket at the given index.
	buckets 		-	Returns all Buckets.
	bucketIndex 	-	Returns the index of the specified bucket.
	findLink		-	Takes as input another Node and finds the Link between this Node that,
						if one exists. If none exists returns None.

	There are functions which modify Nodes by linking them or by setting heirarchy attributes:

	addLink			-	Takes as input another Node as well as the index of the Bucket on this Node
						and the index of the Bucket on the other Node. Links them.

	setParent		-	Sets the parent of this Node to the reference given.

	There are additional functions which create modified copies of Nodes, listed below.
	These may be called only if the node is parentless.

	modify			-	Creates a copy of this Node with the provided Tensor instead. The copy is
	 					one level up in the heirarchy, such that the copy is the parent of this Node.
	 					This also properly links the copy to the rest of the Network. Also takes as
	 					input a boolean indicating whether or not to preserve the compressed status of
	 					Links.
	trace			-	Searches for indices which are linked to one another and produces a new Node
						with them traced out. The new Node is then the parent of this Node.
	merge 			-	Takes as input another Node and merges this Node with it. The net result is that
						a new node is added to the network. This Node and the other are left intact, with
						all of their Links preserved. The rest of the network's Nodes will have Links both
						to these and to the new Node, with Links to the new Node appearing later in their
						Link lists.
	linkMerge		-	Searches for linked Nodes at the top level which have multiple Links between them and
						this one and produces a new pair of nodes as their parents with the Links merged into
						a single higher-dimensional Link.


	Finally, nodes may be deleted:

	delete			-	Delete the Node and all associated Links. Recursively deletes all parents.

	TODO:	Implement tostr method for this and for network so that printing can be sensible.
	'''
	def __init__(self, tens, network, children=[]):
		self.__tensor = tens
		self.__network = network
		self.__id = self.__network.nextID()
		self.__parent = None
		self.__children = children
		self.__network.registerNode(self)
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

	def findLink(self, other):
		for b in self.__buckets:
			for i in range(b.numLinks()):
				if b.otherNode(i) == other:
					return b.link(i)
		return None

	def tensor(self):
		return self.__tensor

	def bucketIndex(self, b):
		return self.__buckets.index(b)

	def linksConnecting(self, other):
		links = []
		for b in self.__buckets:
			if b.otherNode(-1) == other:
				links.append(b.link(-1))
		return links

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

	def modify(self, tens, preserveCompressed = False, delBuckets=[]):
		# delBuckets must have length equal to len(self.tensor().shape()) - len(tens.shape())
		n = Node(tens,self.__network,children=[self])
		counter = 0
		for i,b in enumerate(self.__buckets):
			if i not in delBuckets:
				if b.linked():
					if preserveCompressed:
						n.addLink(b.otherNode(-1),counter,b.otherNode(-1).bucketIndex(b.otherBucket(-1)),compressed=b.link(-1).compressed())
					else:
						n.addLink(b.otherNode(-1),counter,b.otherNode(-1).bucketIndex(b.otherBucket(-1)),compressed=False)
				counter += 1

		return n

	def trace(self):
		for b in self.__buckets:
			if b.linked():
				otherBucket = b.otherBucket(-1)
				otherNode = otherBucket.node()
				if otherNode == self:
					ind0 = self.bucketIndex(b)
					ind1 = self.bucketIndex(otherBucket)
					newT = self.__tensor.trace(ind0,ind1)
					n = self.modify(newT, preserveCompressed = False, delBuckets=[ind0,ind1])
					n.trace() # Keep going until there are no more repeated indices to trace.
					return

	def linkMerge(self,compress=False):
		c = self.connectedHigh()

		for n in c:
			links = self.linksConnecting(n)
			if len(links) > 1:
				n1, n2 = mergeLinks(self, n)
				n1.linkMerge(compress=compress)
		return

	def merge(self, other):
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
	
