from tensor import Tensor
from linkbucket import Link, Bucket
from tensorNetworkNode import Node

class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods for manipulating those Nodes.

	Networks have the following functions:

	registerLink	-	Registers the given Link.
	registerNode	-	Registers the given Node.
	deregisterLink	-	Deregisters the given Link.
	deregisterNode	-	Deregisters the given Node.
	nextID			-	Returns the next ID number.

	
	
	'''


	def __init__(self):
		self.__nodes = set()
		self.__allLinks = set()
		self.__idDict = {}
		self.__idCounter = 0

	def registerLink(self, link):
		self.__allLinks.add(link)

	def deregisterLink(self, link):
		self.__allLinks.remove(link)

	def registerNode(self, node):
		self.__nodes.add(node)

	def deregisterNode(self, node):
		self.__nodes.remove(node)

	def nextID(self):
		idd = self.__idCounter
		self.__idCounter += 1
		return idd

