from tensor import Tensor
from link import Link
from bucket import Bucket
from node import Node
import numpy as np
from compress import compress

class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods for manipulating those Nodes.

	Networks have the following functions:

	registerLink		-	Registers the given Link.
	registerNode		-	Registers the given Node. Also handles tracking top level Links and Nodes.
	deregisterLink		-	Deregisters the given Link.
	deregisterNode		-	Deregisters the given Node. Also handles tracking top level Links and Nodes.
	nextID				-	Returns the next ID number.
	addNodeFromArray	-	Takes as input an array and constructs a Tensor and Node around it,
							then adds the Node to this Network.
	trace				-	Trace trivial loops in all top level Nodes.
	merge				-	Performs the next best merger based on entropy heuristics.
	topLevelNodes 		-	Returns all top level Nodes.
	topLevelLinks 		-	Returns all Links between top-level Nodes.
	nodes 				-	Returns all nodes
	size				-	Returns the size of the Network
	topLevelSize		-	Returns the top-leve size of the Network
	largestTensor		-	Returns the shape of the largest Tensor.
	largestTopLevelTensor	-	Returns the shape of the largest top-level Tensor.

	Note that the logic for keeping track of top level nodes requires that
	nodes be deregistered from the top-down. This is in keeping with the notion
	that the Network should always be valid (there shouldn't be missing interior
	levels in the heirarchy). Links may be deregistered in any fashion.
	'''


	def __init__(self):
		self.__nodes = set()
		self.__topLevelNodes = set()
		self.__allLinks = set()
		self.__idDict = {}
		self.__idCounter = 0

	def size(self):
		s = 0

		for n in self.__nodes:
			s += n.tensor().array().size

		return s

	def topLevelSize(self):
		s = 0

		for n in self.__topLevelNodes:
			s += n.tensor().array().size

		return s

	def largestTensor(self):
		s = 0
		sh = None

		for n in self.__nodes:
			if n.tensor().array().size > s:
				s = n.tensor().array().size
				sh = n.tensor().shape()

		return sh

	def largestTopLevelTensor(self):
		s = 0
		sh = None

		for n in self.__topLevelNodes:
			if n.tensor().array().size > s:
				s = n.tensor().array().size
				sh = n.tensor().shape()

		return sh

	def registerLink(self, link):
		self.__allLinks.add(link)

	def deregisterLink(self, link):
		self.__allLinks.remove(link)

	def registerNode(self, node):
		self.__nodes.add(node)
		self.__topLevelNodes.add(node)

		children = node.children()
		for c in children:
			self.__topLevelNodes.remove(c)

	def deregisterNode(self, node):
		self.__nodes.remove(node)
		self.__topLevelNodes.remove(node)

		children = node.children()
		for c in children:
			self.__topLevelNodes.add(c)

	def nodes(self):
		return self.__nodes

	def nextID(self):
		idd = self.__idCounter
		self.__idCounter += 1
		return idd

	def topLevelNodes(self):
		return self.__topLevelNodes

	def topLevelLinks(self):
		n = self.__topLevelNodes
		links = set()
		for link in self.__allLinks:
			if link.bucket1().node() in n and link.bucket2().node() in n:
				links.add(link)
		return links

	def addNodeFromArray(self, arr):
		t = Tensor(arr.shape,arr)
		n = Node(t,self)
		return n

	def trace(self):
		nodes = list(self.topLevelNodes())

		for n in nodes:
			n.trace()

	def linkMerge(self,compress=False):
		done = set()

		while len(done) < len(self.__topLevelNodes):
			n = list(self.__topLevelNodes.difference(done))[0]
			merged, otherNode = n.linkMerge(compress=compress)
			if not merged:
				done.add(n)

	def merge(self):
		# This logic might make more sense being handled by the Link.
		links = list(self.topLevelLinks())

		s = [link.mergeEntropy(reduction=0.5) for link in links]

		ind = np.argmin(s)

		link = links[ind]

		link.bucket1().topNode().merge(link.bucket2().topNode())

	def compress(self,eps=1e-12):
		compressed = set()

		for link in self.topLevelLinks():
			if link.compressed():
				compressed.add(link)

		while len(compressed) < len(self.topLevelLinks()):
			todo = self.topLevelLinks().difference(compressed)
			todo = list(todo)
			link, _, _ = compress(todo[0],eps=eps)
			compressed.add(link)
