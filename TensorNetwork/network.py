from tensor import Tensor
from link import Link
from bucket import Bucket
from node import Node
import numpy as np
from compress import compress
from priorityQueue import PriorityQueue

class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods for manipulating those Nodes.

	Networks have the following functions:

	registerLink		-	Registers the given Link.
	registerNode		-	Registers the given Node. Also handles tracking top level Links and Nodes.
	deregisterLink		-	Deregisters the given Link.
	deregisterNode		-	Deregisters the given Node. Also handles tracking top level Links and Nodes.
	registerLinkTop		-	Registers the Link as being in the top level.
	deregisterLinkTop	-	Deregisters the Link from the top level.
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
	topView				-	Takes as input a list of bottom-level Nodes and returns the highest-level
							representation of the Network which includes these Nodes. This is constructed
							by recursively adding the highest-level neighbor of a Node currently in the set
							of interest which is not a parent of any Node in the set of interest until the
							new Network is constructed. 

	Note that the logic for keeping track of top level nodes requires that
	nodes be deregistered from the top-down. This is in keeping with the notion
	that the Network should always be valid (there shouldn't be missing interior
	levels in the heirarchy). Links may be deregistered in any fashion.
	'''


	def __init__(self):
		self.__nodes = set()
		self.__topLevelNodes = set()
		self.__allLinks = set()
		self.__topLevelLinks = set()
		self.__sortedLinks = PriorityQueue()
		self.__idDict = {}
		self.__idCounter = 0

	def size(self):
		s = 0

		for n in self.__nodes:
			s += n.tensor().size()

		return s

	def topLevelSize(self):
		s = 0

		for n in self.__topLevelNodes:
			s += n.tensor().size()

		return s

	def largestTensor(self):
		s = 0
		sh = None

		for n in self.__nodes:
			ss = n.tensor().size()
			if ss > s:
				s = ss
				sh = n.tensor().shape()

		return sh

	def largestTopLevelTensor(self):
		s = 0
		sh = None

		for n in self.__topLevelNodes:
			ss = n.tensor().size()
			if ss > s:
				s = ss
				sh = n.tensor().shape()

		return sh

	def registerLink(self, link):
		self.__allLinks.add(link)
		self.__topLevelLinks.add(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

	def deregisterLink(self, link):
		self.__allLinks.remove(link)
		self.__topLevelLinks.remove(link)	# We should only ever remove top-level Links
		self.__sortedLinks.remove(link)

	def deregisterLinkTop(self, link):
		self.__topLevelLinks.remove(link)
		self.__sortedLinks.remove(link)

	def registerLinkTop(self, link):
		self.__topLevelLinks.add(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

	def updateSortedLinkList(self, link):
		self.__sortedLinks.remove(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

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
		return self.__topLevelLinks

	def topView(self, nodes):
		# nodes must be a set
		raise NotImplementedError

	def addNodeFromArray(self, arr):
		t = Tensor(arr.shape,arr)
		n = Node(t,self, Buckets=[Bucket(i,self) for i in range(len(arr.shape))])
		return n

	def trace(self):
		nodes = list(self.topLevelNodes())

		for n in nodes:
			n.trace()

	def linkMerge(self,compress=False):
		done = set()

		while len(done) < len(self.__topLevelNodes):
			n = list(self.__topLevelNodes.difference(done))[0]
			nn = n.linkMerge(compress=compress)
			done.add(nn)

	def merge(self, mergeL=True, compress=True):
		# This logic might make more sense being handled by the Link.
		links = list(self.topLevelLinks())

		link = self.__sortedLinks.pop()

		link.bucket1().topNode().merge(link.bucket2().topNode(), mergeL=mergeL, compress=compress)

	def compress(self,eps=1e-4):
		compressed = set()

#		for link in self.topLevelLinks():
#			if link.compressed():
#				compressed.add(link)

		while len(compressed) < len(self.topLevelLinks()):
			todo = self.topLevelLinks().difference(compressed)
			todo = list(todo)
			link, _, _ = compress(todo[0],eps=eps)
			compressed.add(link)
