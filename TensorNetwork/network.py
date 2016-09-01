from tensor import Tensor
from link import Link
from bucket import Bucket
from node import Node
import numpy as np
from compress import compress
from priorityQueue import PriorityQueue

class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods
	for manipulating those Nodes.
	'''

	def __init__(self):
		'''
		Method for initializing an empty Network.
		'''
		self.__nodes = set()
		self.__topLevelNodes = set()
		self.__allLinks = set()
		self.__topLevelLinks = set()
		self.__cutLinks = set()
		self.__sortedLinks = PriorityQueue()
		self.__idDict = {}
		self.__idCounter = 0

	def __str__(self):
		'''
		Returns string representations of all top-level Nodes in the Network.
		The are no guarantees of the order in which this is done.
		'''
		s = 'Network\n'
		for n in self.__topLevelNodes:
			s = s + str(n) + '\n'
		return s

	def size(self):
		'''
		Returns the sum of the sizes of all Tensors in the Network.
		'''
		return sum(n.tensor().size() for n in self.__nodes)

	def topLevelSize(self):
		'''
		Returns the sum of the sizes of all Tensors belonging to top-level Nodes in the Network.
		'''
		return sum(n.tensor().size() for n in self.__topLevelNodes)

	def nodes(self):
		'''
		Returns a copy of the set of all Nodes in the Network.
		'''
		return set(self.__nodes)

	def topLevelNodes(self):
		'''
		Returns a copy of the set of all top-level Nodes in the Network.
		'''
		return set(self.__topLevelNodes)

	def topLevelLinks(self):
		'''
		Returns a copy of the set of all top-level Links in the Network.
		'''
		return set(self.__topLevelLinks)

	def largestTensor(self):
		'''
		Returns the Node containing the largest Tensor in the Network.
		'''
		sizeGetter = lambda n: n.tensor().size()
		return max(self.__nodes, key=sizeGetter)

	def largestTopLevelTensor(self):
		'''
		Returns the top-level Node with the largest Tensor in the Network.
		'''
		sizeGetter = lambda n: n.tensor().size()
		return max(self.__topLevelNodes, key=sizeGetter)

	def topLevelRepresentation(self):
		'''
		Returns the tensor product of all top-level Tensors along with
		a list of corresponding Buckets in the same order as the indices.
		'''
		arr = np.array([1.])
		logS = 0
		bucketList = []

		for n in self.__topLevelNodes:
			arr = np.tensordot(arr, n.tensor().array(), axes=0)
			logS += n.logScalar()
			for b in n.buckets():
				bucketList.append(b)

		return arr[0], logS, bucketList

	def registerLink(self, link):
		'''
		Registers a new Link in the Network.
		This should only be called when a Link is created.
		'''
		assert link not in self.__allLinks
		assert link not in self.__topLevelLinks

		self.__allLinks.add(link)
		self.registerLinkTop(link)

	def deregisterLink(self, link):
		'''
		De-registers a Link from the Network.
		This should only be used when deleting a Link.
		'''
		assert link in self.__allLinks
		assert link in self.__topLevelLinks or link in self.__cutLinks

		self.__allLinks.remove(link)
		if link in self.__topLevelLinks:
			self.__topLevelLinks.remove(link)
		else:
			self.__cutLinks.remove(link)
		self.__sortedLinks.remove(link)

	def registerLinkTop(self, link):
		'''
		Registers a Link in the Network as being top-level.
		This is called by registerLink, and hence is used when a Link is created.
		It is also called when a Link is deleted (so that the children of that Link
		may become top-level).
		'''
		assert link not in self.__topLevelLinks
		assert link.bucket1().topNode() in self.__topLevelNodes
		assert link.bucket2().topNode() in self.__topLevelNodes

		self.__topLevelLinks.add(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

	def deregisterLinkTop(self, link):
		'''
		De-registers a Link in the Network from being top-level.
		This should be called only when a Link is traced out,
		compressed, or merged with another Link.
		'''
		assert link in self.__allLinks
		assert link in self.__topLevelLinks

		self.__topLevelLinks.remove(link)
		self.__sortedLinks.remove(link)


	def registerLinkCut(self, link):
		'''
		Registers a link as having been cut and de-registers is from the top-level.
		This should only be called when a Link is cut.
		This occurs when, upon compression, the Link is reduced to bond dimension 1.
		'''
		assert link not in self.__cutLinks
		assert link in self.__topLevelLinks
		assert link in self.__allLinks

		self.__cutLinks.add(link)
		self.deregisterLinkTop(link)

	def deregisterLinkCut(self, link):
		'''
		De-registers a Link from being cut and adds it to the top-level.
		This is called only when a Node directly above one on either side of
		a cut Link is deleted, as that indicates that the bond ought to be
		active once more (the compression resulting in it being cut has been
		undone).
		'''
		assert link in self.__cutLinks
		assert link in self.__allLinks
		assert link not in self.__topLevelLinks

		self.__cutLinks.remove(link)
		self.registerLinkTop(link)

	def updateSortedLinkList(self, link):
		'''
		Updates the position of the given Link in the priority queue of Links
		to be contracted.
		'''
		self.__sortedLinks.remove(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

	def registerNode(self, node):
		'''
		Registers a new Node in the Network.
		This should only be called when registering a new Node.
		'''
		assert node not in self.__nodes
		assert node not in self.__topLevelNodes
		assert len(set(node.children()).intersection(self.__topLevelNodes)) == len(node.children())

		self.__nodes.add(node)
		self.__topLevelNodes.add(node)

		children = node.children()
		for c in children:
			self.__topLevelNodes.remove(c)

		assert len(set(node.children()).intersection(self.__topLevelNodes)) == 0


	def deregisterNode(self, node):
		'''
		De-registers a Node from the Network.
		This should only be called when deleting a Node.
		This also handles updating the link registration
		in the event that the Node was formed from contracting
		a Link.
		'''

		self.__nodes.remove(node)
		self.__topLevelNodes.remove(node)

		children = node.children()

		self.__topLevelNodes.update(children)

		for b in node.buckets():
			assert b.topNode() == node

		if len(children) == 2:
			links = children[0].linksConnecting(children[1])
			for l in links:
				self.registerLinkTop(l)

	def nextID(self):
		'''
		Returns the next unused ID number in the Network.
		'''
		idd = self.__idCounter
		self.__idCounter += 1
		return idd


	def addNodeFromArray(self, arr):
		'''
		Takes as input an array and constructs a Tensor and Node around it,
		then adds the Node to this Network.
		'''
		t = Tensor(arr.shape, arr)
		return Node(t, self, Buckets=[Bucket(self) for _ in range(len(arr.shape))])

	def trace(self):
		'''
		Traces over all Nodes in the Network.
		'''
		nodes = list(self.topLevelNodes())
		for n in nodes:
			n.trace()

	def merge(self, mergeL=True, compressL=True, eps=1e-4):
		'''
		Performs the next best merger (contraction) between Nodes based on entropy heuristics.
		The Nodes must be linked to one another.

		This method takes three keyword arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''

		link = self.__sortedLinks.pop()

		n1 = link.bucket1().topNode()
		n2 = link.bucket2().topNode()

		n1.merge(n2, mergeL=mergeL, compressL=compressL, eps=eps)


	def linkMerge(self, compressL=False, eps=1e-4):
		'''
		This method checks all Nodes for potential Link mergers and performs any it finds.
		This method takes two keyword arguments:
			compressL	-	Attempts to compress all Links (if any) resulting from a Link merger.
			eps			-	The accuracy of the compression to perform.
		'''
		done = set()
		todo = set(self.__topLevelNodes)

		while len(todo) > 0:
			n = todo.pop()
			nn, d, new = n.linkMerge(compressL=compressL, eps=eps)

			todo = todo.difference(d)
			todo = todo | new

			done.add(nn)

	def contract(self, mergeL=True, compressL=True, eps=1e-4):
		'''
		This method contracts the Network to a minimal representation.

		This method takes three keywork arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''
		self.trace()

		counter = 0
		while self.__sortedLinks.length > 0:
			self.merge(mergeL=True, compressL=True)

			if counter%20 == 0:
				t = self.largestTopLevelTensor()
				print(len(self.topLevelNodes()),self.topLevelSize(), t.tensor().shape())
			counter += 1

	def compressLinks(self, eps=1e-4):
		'''
		This method attempts to compress all top-level Links in the Network.

		This method takes one keyword argument:
			eps	-	The accuracy of the compression to perform.
		'''

		compressed = set()

		while len(compressed) < len(self.topLevelLinks()):
			todo = self.topLevelLinks().difference(compressed)
			todo = list(todo)
			link, _, _ = compress(todo[0], eps=eps)
			compressed.add(link)

	def view(self, nodes, mergeL=True, compressL=True, eps=1e-4):
		'''
		This method calculates the effective Tensor represented by the Network connecting to the given Nodes.

		The way we do this is straightforward: we go through marking Nodes as rejected
		as if we were deleting them. This includes following compression/merger bonds
		where appropriate. We then copy the highest-level Nodes remaining into a new
		Network and contract it.

		This method takes three keywork arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''
		todo = set(nodes)
		rejected = set()

		while len(todo) > 0:
			n = todo.pop()

			if n not in rejected:
				rejected.add(n)
				if n.parent() is not None:
					todo.add(n.parent())

				if len(n.children()) == 1:
					for b in n.buckets():
						link = b.link()
						bo = link.otherBucket(b)
						no = bo.bottomNode()
						if len(no.children()) == 1:
							todo.add(no)

		new1 = set(self.__nodes).difference(rejected)
		new2 = set(self.__nodes).difference(rejected)

		for n in new1:
			if n.parent() in new1:
				new2.remove(n)

		# Build the new Network
		nn = Network()

		# Build the new Nodes
		oldIDs = {}	#	Returns the old Node corresponding to the given new ID.
		newIDs = {}	#	Returns the new Node corresponding to the given old ID.

		newNodes = set()

		for n in new2:
			n1 = nn.addNodeFromArray(n.tensor().array())

			oldIDs[n1.id()] = n
			newIDs[n.id()] = n1

			newNodes.add(n1)


		# Link up new Nodes
		done = set()

		for n1 in new2:
			n1new = newIDs[n1.id()]
			for ind0, b in enumerate(n1.buckets()):
				if b.linked():
					otherB = b.otherBucket()
					intersection = set(otherB.nodes()).intersection(new2)
					if len(intersection) > 0:
						n2 = intersection.pop()
						ind1 = n2.buckets().index(otherB)
						n2new = newIDs[n2.id()]
						if (n1new.id(), ind0, n2new.id(), ind1) not in done:
							n1new.addLink(n2new, ind0, ind1)
							done.add((n1new.id(), ind0, n2new.id(), ind1))
							done.add((n2new.id(), ind1, n1new.id(), ind0))

		# Contract new Network
		nn.contract(mergeL=mergeL, compressL=compressL, eps=eps)

		# Build contracted Tensor and bucketList
		arr = np.array([1.])
		bucketList = []

		for n in nn.topLevelNodes():
			arr = np.tensordot(arr, n.tensor().array(), axes=0)
			for b in n.buckets():
				nb = b.bottomNode()
				ind = nb.bucketIndex(b)
				oldNode = oldIDs[nb.id()]
				bucketList.append(oldNode.buckets()[ind].otherBucket())

		return nn, arr[0], bucketList





