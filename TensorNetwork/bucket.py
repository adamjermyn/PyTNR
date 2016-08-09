class Bucket:
	'''
	A Bucket is a means of externally referencing an index of a tensor which handles the way in which
	the tensor is linked to other tensors.

	Each Bucket references exactly one index, but may contain multiple Links between that index and others.
	This allows a given tensor to be part of a heirarchical network, wherein nodes may be merged while
	retaining information about the unmerged structure. To accomodate this, each Bucket contains a list
	of Links. When two nodes are merged, the old links remain, while a new Link to the merged object
	is added to the end of the Link list.

	Buckets have the following functions:

	node 		-	Returns the Node this Bucket belongs to.
	index 		-	Returns the index of the Node's Tensor this Bucket refers to.
	network 	-	Returns the TensorNetwork this Bucket belongs to.
	numLinks	-	Returns the number of Links this Bucket has.
	links 		-	Returns all Links this Bucket has.
	link 		-	Takes as input an integer specifying the index of the Link of interest and returns
					that link.
	linked		-	Returns True if the Bucket is linked. Returns False otherwise.
	otherBucket	-	Takes as input an integer specifying the index of the Link of interest and returns
					the Bucket on the other side of that Link.
	otherNode	-	Takes as input an integer specifying the index of the Link of interest and returns
					the Node on the other side of that Link.
	addLink		-	Takes as input a Link and appends it to the end of the Link list.
	removeLink	-	Removes a Link from the Link list. Raises a ValueError if the Link is not present.
	'''

	def __init__(self, node, index, network):
		self.__node = node
		self.__index = index
		self.__network = network

		self.__links = []

	def node(self):
		return self.__node

	def index(self):
		return self.__index

	def network(self):
		return self.__network

	def numLinks(self):
		return len(self.__links)

	def links(self):
		return self.__links

	def link(self, index):
		return self.__links[index]

	def linked(self):
		return (len(self.__links) > 0)

	def otherBucket(self, index):
		b = self.__links[index].bucket1()
		if b == self:
			b = self.__links[index].bucket2()
		return b

	def otherNode(self, index):
		return self.otherBucket(index).node()

	def addLink(self, link):
		self.__links.append(link)

	def removeLink(self, link):
		if link in self.__links:
			self.__links.remove(link)
		else:
			raise ValueError