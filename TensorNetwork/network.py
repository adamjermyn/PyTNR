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

		link.bucket1().topNode().merge(link.bucket2().topNode(), mergeL=mergeL, compress=compressL, eps=eps)


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
			nn, d, new = n.linkMerge(compress=compressL, eps=eps)

			todo = todo.difference(d)
			todo = todo | new

			done.add(nn)


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
