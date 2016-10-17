
class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods
	for manipulating those Nodes.
	'''

	def __init__(self):
		self.nodes = set()
		self.buckets = set()
		self.internalBuckets = set()
		self.externalBuckets = set()
		self.size = 0

	def __str__(self):
		s = 'Network\n'
		for n in self.nodes:
			s = s + str(n) + '\n'
		return s

	def addNode(self, node):
		'''
		Registers a new Node in the Network.
		This should only be called when registering a new Node.
		All links between this node and other nodes in this network
		must already exist, so in that sense adding the Node ought to
		be the last thing that is done.
		'''
		assert node not in self.nodes

		self.nodes.add(node)
		self.buckets.update(node.buckets)
		for b in node.buckets:
			if b.linked and b.otherNode in self.nodes:
				self.internalBuckets.add(b)
				self.internalBuckets.add(b.otherBucket)
				if b.otherBucket in self.externalBuckets:
					self.externalBuckets.remove(b.otherBucket)
			else:
				self.externalBuckets.add(b)

	def removeNode(self, node):
		'''
		De-registers a Node from the Network.
		This should only be called when deleting a Node.
		This also handles updating the link registration
		in the event that the Node was formed from contracting
		a Link.
		'''
		assert node in self.nodes

		self.nodes.remove(node)
		self.buckets = self.buckets.difference(node.buckets)
		for b in node.buckets:
			if b in self.internalBuckets:
				self.internalBuckets.remove(b)
				if b.otherBucket in self.internalBuckets:
					self.externalBuckets.add(b.otherBucket)
			if b in self.externalBuckets:
				self.externalBuckets.remove(b)

	def mergeNodes(self, n1, n2):
		'''
		Merges the specified Nodes.
		'''
		n = n1.mergeNodes(n2)

		self.addNode(n)
		self.removeNode(n1)
		self.removeNode(n2)

		return n