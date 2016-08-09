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
	topLevelLinks 		-	Returns all top level Links.
	nodes 				-	Returns all nodes

	Note that the logic for keeping track of top level nodes requires that
	nodes be deregistered from the top-down. This is in keeping with the notion
	that the Network should always be valid (there shouldn't be missing interior
	levels in the heirarchy).

	'''


	def __init__(self):
		self.__nodes = set()
		self.__topLevelNodes = set()
		self.__allLinks = set()
		self.__topLevelLinks = set()
		self.__idDict = {}
		self.__idCounter = 0

	def registerLink(self, link):
		self.__allLinks.add(link)
		self.__topLevelLinks.add(link)

	def deregisterLink(self, link):
		self.__allLinks.remove(link)
		self.__topLevelLinks.remove(link)

	def registerNode(self, node):
		self.__nodes.add(node)
		self.__topLevelNodes.add(node)

		children = node.children()
		for c in children:
			print c.id()
			self.__topLevelNodes.remove(c)
			buckets = c.buckets()
			for b in buckets:
				if b.linked() and b.link(-1) in self.__topLevelLinks:
					self.__topLevelLinks.remove(b.link(-1))

	def deregisterNode(self, node):
		self.__nodes.remove(node)

		children = node.children()
		for c in children:
			self.__topLevelNodes.add(c)
			buckets = c.buckets()
			for b in buckets:
				if b.linked():
					self.__topLevelLinks.add(b.link(-1))

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

	def addNodeFromArray(self, arr):
		t = Tensor(arr.shape,arr)
		n = Node(t,self)
		return n

	def trace(self):
		nodes = list(self.__topLevelNodes)

		for n in nodes:
			n.trace()

	def merge(self):
		links = list(self.__topLevelLinks)

		s = [link.mergeEntropy() for link in links]

		ind = np.argmin(s)

		link = links[ind]

		link.bucket1().node().merge(link.bucket2().node())

	def compress(self,tol=1e-4):
		compressed = set()

		for link in self.__topLevelLinks:
			if link.compressed():
				compressed.add(link)

		while len(compressed) < len(self.__topLevelLinks):
			todo = self.__topLevelLinks.difference(compressed)
			todo = list(todo)
			print 'lt',len(todo)
			link = compress(todo[0],tol)
			compressed.add(link)
			