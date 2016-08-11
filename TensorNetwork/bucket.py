class Bucket:
	'''
	A Bucket is a means of externally referencing an index of a tensor which handles the way in which
	the tensor is linked to other tensors.

	Each Bucket references exactly one index and up to one link, but a Bucket may be associated with
	multiple Nodes (and hence Tensors). More specifically, in the Network we build a heirarchical structure
	in which Tensors are constructed based on those in lower levels. A Bucket may be associated with a line
	of parents/children in this structure. This indicates that the physical meaning of the relevant index
	has gone unchanged through the transformations that produce this structure.

	The insistence that no more than one Link exist between Buckets reflects the physical nature of the Link.

	When two Nodes are merged, the merged object becomes their parent and acquires all of their Buckets, sans
	the ones which were traced over in the merger. When a Link merger or Link compression occurs, new Buckets
	must be created for the relevant indices, as their physical meaning has changed with the associated change
	of basis. When a trace occurs no new Buckets are required.

	The order of Nodes in the __nodes attribute respects the parent/child relationship, such that the
	highest-level Node (the one with no parents in the set) is the final element and the lowest-level Node
	(the one with no children in the set) is the first element.

	Buckets have the following functions:

	link 		-	Returns the Link, if any, associated with this Bucket. Returns None if there is no Link.
	linked		-	Returns True if the Bucket is linked. Returns False otherwise.
	nodes 		-	Return all Nodes associated with this Bucket.
	node 		-	Return the Node at the given index.
	numNodes	-	Returns the number of Nodes associated with this Bucket.
	topNode		-	Returns the top-Level Node associated with this Bucket.
	network 	-	Returns the TensorNetwork this Bucket belongs to.
	otherBucket	-	Returns the Bucket on the other side of the Link, if this Bucket is linked. If it is not
					returns None.
	otherNodes	-	Returns the list of Nodes associated with the other Bucket associated with the Link
					associated with this Bucket.
	otherNode	-	Takes as input an integer specifying the index of the Node of interest and returns the
					Node associated with the other Bucket of the Link associated with this Bucket at that
					index. If there is no Link raises a ValueError.
	otherTopNode	-	Returns the highest-level Node associated with the other Bucket associated with the Link
						associated with this Bucket.
	numOtherNodes	-	Returns the number of Nodes associated with the other Bucket associated with the
						Link associated with this Node. If there is no Link reutnrs 0.
	addNode		-	Takes as input a Node and adds it to the end of the list of Nodes.
	setLink		-	Takes as input a Link and sets it as the one associated with this Bucket.
	removeLink	-	Removes the Link associated with this Bucket.
					Raises a ValueError if the Link is not present.
	'''

	def __init__(self, index, network):
		self.__nodes = []
		self.__network = network
		self.__link = None
		self.__otherBucket = None

	def link(self):
		return self.__link

	def linked(self):
		return (self.__link is not None)

	def node(self, index):
		return self.__nodes[index]

	def nodes(self):
		return self.__nodes

	def numNodes(self):
		return len(self.__nodes)

	def topNode(self):
		return self.__nodes[-1]

	def network(self):
		return self.__network

	def otherBucket(self):
		return self.__otherBucket

	def otherNodes(self):
		if not self.linked():
			raise ValueError
		return self.__otherBucket.nodes()

	def otherNode(self, index):
		return self.otherNodes()[index]

	def otherTopNode(self):
		return self.otherNode(-1)

	def numOtherNodes(self):
		if not self.linked():
			raise ValueError
		return self.otherBucket().numNodes()

	def addNode(self, node):
		self.__nodes.append(node)
		if self.linked():
			self.__link.updateMergeEntropy()

	def setLink(self, link):
		self.__link = link
		self.__otherBucket = link.otherBucket(self)
			# A condition of this logic for otherBucket is that we never change which Bucket
			# a Link points to once we set it. This is fine, as there are no modifier methods
			# in the Link class for the Buckets it points to.

	def removeLink(self, link):
		self.__link = None
		self.__otherBucket = None
