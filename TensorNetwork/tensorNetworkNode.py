from linkbucket import Link, Bucket
from tensor import Tensor

class Node:
	'''
	A Node object stores the Buckets corresponding to a given Tensor.
	It also provides helper methods allowing for easy access to neighbors.

	Node objects have the following functions which do not modify them:

	id 				-	Returns the id number of the Node. These numbers are unique within a network.
	children		-	Returns the children Nodes (if any) which merged to form this Node.
	parent			-	Returns the Node (if any) which this merges to form.
	connected		-	Returns the Nodes this one is connected to.
	connectedHigh	-	Returns the Nodes this one is connected to, giving the highest-level list possible.
						That is, the list such that no element of the set has a parent which is connected
						to this Node, and such that all Nodes connected to this one are children to some
						degree of an element in the list.
	tensor 			-	Returns the Tensor underlying this Node.
	bucketIndex 	-	Returns the index of the specified bucket.

	There are functions which modify Nodes by linking them:

	addLink			-	Takes as input another Node as well as two indices and creates a Link between
						this Node and the other, this one on the first index and the other on the second.

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

	TODO:
	1. Implement merge.
	2. Implement linking function (addLink).
	'''
	def __init__(self, tens, network, children=[]):
		self.__tensor = tens
		self.__network = network
		self.__network.registerTensor(self)
		self.__id = self.__network.nextId()
		self.__children = children
		self.__parent = None
		self.__buckets = [Bucket(self,i,self.__network) for i in range(len(self.__tensor.shape()))]


	def id(self):
		return self.__id

	def children(self):
		return self.__children

	def parent(self):
		return self.__parent

	def connected(self):
		c = []
		for b in self.__buckets:
			for i in range(len(b.numLinks())):
				c.append(b.otherNode(i))
		return c

	def connectedHigh(self):
		c = []
		for b in self.__buckets:
			c.append(b.otherNode(b.numLinks()-1))
		return c

	def tensor(self):
		return self.__tensor

	def bucketIndex(self, b):
		return self.__buckets.index(b)

	def delete(self):
		if self.__parent is not None:
			self.__parent.delete()

		for b in self.__buckets:
			for l in b.links():
				l.delete()

		for c in self.children():
			c.parent = None

		self.__network.deregisterTensor(self)

		del self.__buckets
		del self

	def trace(self):
		for b in self.__buckets:
			otherBucket = b.otherBucket(b.numLinks()-1)
			otherNode = otherBucket.node()
			if otherNode == self:
				ind0 = self.bucketIndex(b)
				ind1 = self.bucketIndex(otherBucket)
				newT = self.__tensor.trace(ind0,ind1)
				n = Node(newT,self.__network,children=self)
				self.__parent = n
				counter = 0
				for bb in self.__buckets:
					if bb.linked() and bb != b and bb != otherBucket:
						# TODO: Make addLink signature below conform to specification above.
						n.addLink(counter,bb.otherBucket(bb.numLinks()-1))
				n.trace()
				return

	def merge(self):
		raise NotImplementedError


	

'''

	def swapIndices(self, i, j):
		self.buckets[i], self.buckets[j] = self.buckets[j], self.buckets[i]
		self.array = np.swapaxes(self.array,i,j)

	def mergeLinks(self, other):
		links = self.connected[other]
		if len(links) >= 2:
			lenlinks = len(links)
			for i in range(len(links)):
				ind = None
				if links[i].bucket1.tensor == self:
					ind = self.buckets.index(links[i].bucket1)
				else:
					ind = self.buckets.index(links[i].bucket2)
				self.swapIndices(i,ind)

				ind = None
				if links[i].bucket1.tensor == other:
					ind = other.buckets.index(links[i].bucket1)
				else:
					ind = other.buckets.index(links[i].bucket2)
				other.swapIndices(i,ind)

			self.array = np.reshape(self.array,[-1] + list(self.array.shape[len(links):]))
			other.array = np.reshape(other.array,[-1] + list(other.array.shape[len(links):]))

			for i in range(len(links)-1):
				self.buckets[i].link.delete()

			self.buckets = self.buckets[lenlinks-1:]
			other.buckets = other.buckets[lenlinks-1:]

	def mergeAllLinks(self):
		for t in self.connected.keys():
			self.mergeLinks(t)

	def trace(self, ind0, ind1):
		self.array = np.trace(self.array, axis1=ind0, axis2=ind1)

		b0 = self.buckets[ind0]
		b1 = self.buckets[ind1]

		self.buckets.remove(b0)
		self.buckets.remove(b1)

		del b0
		del b1

	def addLink(self, other, indSelf, indOther, kind='outside'):
		# If kind is outside then indSelf and indOther are assumed to refer to outside (original) indices.
		# Otherwise they are inside indices.
		if kind=='outside':
			for q in range(len(self.buckets)):
				if self.buckets[q].index == indSelf:
					i = q
			for q in range(len(other.buckets)):
				if other.buckets[q].index == indOther:
					j = q
		else:
			i = indSelf
			j = indOther

		if self.buckets[i].link is not None:
			raise ValueError('Error: That bucket is already occupied.')
		if other.buckets[j].link is not None:
			raise ValueError('Error: That bucket is already occupied.')

		if self == other:
			self.trace(i,j)
		else:
			b1 = self.buckets[i]
			b2 = other.buckets[j]

			# Build a link
			b1.makeLink(b2)
			l = b1.link

	def contract(self, other, reshape=True): # There should be just one link
		if other not in self.connected:
			raise ValueError('Tensors not connected!')
		else:
			self.mergeLinks(other)
			link = self.connected[other][0]
			bSelf = link.bucket1
			bOther = link.bucket2

			indSelf = self.buckets.index(bSelf)
			indOther = other.buckets.index(bOther)

			t = np.tensordot(self.array, other.array, axes=((indSelf,),(indOther,)))

			if reshape:
				prodSelf = [self.array.shape[j] for j in range(len(self.array.shape)) if j != indSelf]
				prodOther = [other.array.shape[j] for j in range(len(other.array.shape)) if j != indOther]

				prodSelf = np.prod(prodSelf)
				prodOther = np.prod(prodOther)

				t = np.reshape(t,(prodSelf, prodOther))

			return t
'''