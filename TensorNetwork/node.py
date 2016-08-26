from link import Link
from bucket import Bucket
from tensor import Tensor
from mergeLinks import mergeLinks
from collections import Counter
import numpy as np

class Node:
	'''
	A Node object stores the Buckets corresponding to a given Tensor.
	It also provides helper methods allowing for easy access to neighbors.

	Node objects have the following functions which do not modify them:

	id 				-	Returns the id number of the Node. These numbers are unique within a network.
	children		-	Returns the children Nodes (if any) which merged to form this Node.
	parent			-	Returns the Node (if any) which this merges to form.
	topParent		-	Returns the highest-level ancestor of this Node.
	allNChildren	-	Returns all Nodes for which topParent returns this Node.
	network 		-	Returns the Network this Node belongs to.
	connected		-	Returns the Nodes this one is connected to.
	connectedHigh	-	Returns the Nodes this one is connected to, giving the highest-level list possible.
						That is, the list such that no element of the set has a parent which is connected
						to this Node, and such that all Nodes connected to this one are children to some
						degree of an element in the list.
	tensor 			-	Returns the Tensor underlying this Node.
	logScalar 		-	Returns the log of the scalar component that has been divided out from this Tensor.
	bucket 			-	Returns the Bucket at the given index.
	buckets 		-	Returns all Buckets.
	bucketIndex 	-	Returns the index of the specified bucket.
	findLink		-	Takes as input another Node and finds the Link between this Node that,
						if one exists. If none exists returns None.
	linksConnecting	-	Returns all Links connecting this Node and another (provided as input).

	There are functions which modify Nodes by linking them or by setting heirarchy attributes:

	addLink			-	Takes as input another Node as well as the index of the Bucket on this Node
						and the index of the Bucket on the other Node. Links them. If they are already
						Linked this raises a ValueError.

	setParent		-	Sets the parent of this Node to the reference given.

	There are additional functions which create modified copies of Nodes, listed below.
	These may be called only if the node is parentless.

	modify			-	Creates a copy of this Node with the provided Tensor instead. The copy is
	 					one level up in the heirarchy, such that the copy is the parent of this Node.
	 					This also properly links the copy to the rest of the Network.
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
	'''
	def __init__(self, tens, network, children=[], Buckets=[], logScalar = 0):
		self.__tensor = tens
		self.__logScalar = logScalar + np.log(self.__tensor.makeUnity())
		self.__network = network
		self.__id = self.__network.nextID()
		self.__parent = None
		self.__children = children
		self.__buckets = Buckets
		self.__network.registerNode(self)
		for b in Buckets:
			b.addNode(self)
		for c in self.__children:
			c.setParent(self)

	def id(self):
		return self.__id

	def __str__(self):
		return 'Node with ID: ' + str(self.__id) + '  and tensor shape ' + str(self.__tensor.shape())

	def children(self):
		return self.__children

	def parent(self):
		return self.__parent

	def topParent(self):
		if self.parent() is None:
			return self
		else:
			return self.parent().topParent()

	def allNChildren(self):
		ch = set(self.__children)
		for c in self.__children:
			ch = ch | set(c.allNChildren())
		return ch

	def setParent(self, parent):
		self.__parent = parent

	def network(self):
		return self.__network

	def connected(self):
		c = []
		for b in self.__buckets:
			if b.linked():
				c.extend(b.otherNodes())
		return c

	def connectedHigh(self):
		c = []
		for b in self.__buckets:
			if b.linked():
				c.append(b.otherTopNode())
		return c

	def findLink(self, other):
		for b in self.__buckets:
			if b.linked():
				if other in b.otherNodes():
					return b.link()
		return None

	def linksConnecting(self, other):
		links = []
		for b in self.__buckets:
			if b.linked():
				if other in b.otherNodes():
					links.append(b.link())
		return links

	def tensor(self):
		return self.__tensor

	def logScalar(self):
		return self.__logScalar

	def bucketIndex(self, b):
		return self.__buckets.index(b)

	def bucket(self, i):
		return self.__buckets[i]

	def buckets(self):
		return self.__buckets

	def delete(self):
		'''
		I think the error we're still getting is due to the lack of case handling for
		links which get compressed....
		'''

		print self

		if self.__parent is not None:
			self.__parent.delete()

		for b in self.__buckets:
			if b.linked():
				link = b.link()

				bo = b.otherTopNode()

				numC = len(link.children())

				if b.numNodes() == 1:
					# We only delete a Link if this is the last Node its Bucket connects to.
					link.delete()

					if numC > 0:
						# Means the link we're about to delete is a compressed or merged link.
						# This means we need to delete the Node on the other end too.
						# We have to do it in this order so bo isn't linked to this Nodes
						# (which would cause an infinite loop).
						bo.delete() 
			else:
				b.removeNode() # We only ever delete top-level nodes

		for c in self.children():
			c.setParent(None)

		print '---'
		print self
		self.__network.deregisterNode(self)
		print self
		print 'Successful deletion.'
		print '---'
		
		import sys
		import gc
		print sys.getrefcount(self)
		for i in gc.get_referrers(self):
			print i

	def addLink(self, other, selfBucketIndex, otherBucketIndex, compressed=False, children=[]):
		assert self in self.__network.topLevelNodes()
		assert other in self.__network.topLevelNodes()

		if children is None:
			children = []

		selfBucket = self.bucket(selfBucketIndex)
		otherBucket = other.bucket(otherBucketIndex)

		if selfBucket.linked():
			raise ValueError
		if otherBucket.linked():
			raise ValueError

		l = Link(selfBucket,otherBucket,self.__network,compressed=compressed,children=children)

		selfBucket.setLink(l)
		otherBucket.setLink(l)

		return l

	def modify(self, tens, delBuckets=None, repBuckets=None):
		'''
		len(delBuckets) + len(tens.shape()) - len(newBuckets) == len(self.tensor().shape())
		Creates a copy of this Node with tens as its Tensor.  Omits buckets at indices listed in
		delBuckets. Replaces Buckets at indices listed in repBuckets with new Bucket objects.
		Raises a ValueError if repBuckets and delBuckets contain overlapping elements.
		'''
		if delBuckets is None:
			delBuckets = []

		if repBuckets is None:
			repBuckets = []

		assert self in self.__network.topLevelNodes()
		assert len(set(delBuckets).intersection(set(repBuckets))) == 0
		assert len(delBuckets) + len(tens.shape()) - len(self.tensor().shape()) >= 0

		Buckets = []

		for i,b in enumerate(self.buckets()):
			if i not in delBuckets:
				if i not in repBuckets:
					Buckets.append(b)
				else:
					Buckets.append(Bucket(self.network()))

		n = Node(tens, self.__network, children=[self], Buckets=Buckets, logScalar=self.__logScalar)

		return n

	def trace(self):
		assert self in self.__network.topLevelNodes()

		for b in self.__buckets:
			if b.linked():
				otherBucket = b.otherBucket()
				otherNode = otherBucket.topNode()
				if otherNode == self:
					ind0 = self.bucketIndex(b)
					ind1 = self.bucketIndex(otherBucket)
					newT = self.__tensor.trace(ind0, ind1)
					n = self.modify(newT, delBuckets=[ind0,ind1])
					self.__network.deregisterLinkTop(b.link())
					n = n.trace() # Keep going until there are no more repeated indices to trace.
					return n
		return self

	def linkMerge(self,compress=False):
		assert self.__parent is None

		todo = set()
		done = set()
		new = set()

		c = Counter(self.connectedHigh())

		for n in c:
			if c[n] > 1:
				todo.add(n)

		n1 = self

		while len(todo) > 0:
			n = todo.pop()
			done.add(n)

			n1, n2 = mergeLinks(n1, n, compressLink = compress)

			assert len(n1.children()) == len(n2.children())

			new.add(n2)

		return n1, done, new

	def merge(self, other, mergeL=True, compress=True):
		assert self in self.__network.topLevelNodes()
		assert other in self.__network.topLevelNodes()
		assert self in other.connectedHigh()
		assert other in self.connectedHigh()

		# Find all links between self and other, store their indices, and
		# deregister them from the top level
		links = []
		for i,b in enumerate(self.__buckets):
			if b.linked():
				if b.otherTopNode() == other:
					links.append((i,other.bucketIndex(b.otherBucket())))
					self.__network.deregisterLinkTop(b.link())

		links = zip(*links)

		# Contract along common links
		t = self.__tensor.contract(links[0],other.tensor(),links[1])

		# Build new Node
		Buckets = []
		for b in self.buckets():
			if not b.linked():
				Buckets.append(b)
			elif b.otherTopNode() != other:
				Buckets.append(b)
		for b in other.buckets():
			if not b.linked():
				Buckets.append(b)
			elif b.otherTopNode() != self:
				Buckets.append(b)

		# Build new Node
		n = Node(t,self.__network,children=[self,other], Buckets=Buckets, logScalar = self.logScalar() + other.logScalar())	

		# Trace out any self-loops
		n = n.trace()

		if mergeL:
			# Merge any links that need it. The next line is probably redundant given that trace now returns the top level.
			n = n.topParent()
			n, _, _= n.linkMerge(compress=compress)

		return n
