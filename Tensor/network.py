from node import Node
from link import Link
from compress import compressLink

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
		self.optimizedLinks = set()
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
		assert node.network is None

		node.network = self
		self.nodes.add(node)
		for b in node.buckets:
			self.buckets.add(b)
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

		node.network = None
		self.nodes.remove(node)
		for b in node.buckets:
			self.buckets.remove(b)
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
		links = n1.linksConnecting(n2)
		indices = [[],[]]
		for l in links:
			b1, b2 = l.bucket1, l.bucket2
			if b1.node == n2:
				b1, b2 = b2, b1
			assert b1.node == n1
			assert b2.node == n2
			indices[0].append(b1.index)
			indices[1].append(b2.index)

		t = n1.tensor.contract(indices[0], n2.tensor, indices[1])

		buckets = []
		for b in n1.buckets:
			if not b.linked or b.otherBucket not in n2.buckets:
				buckets.append(b)
		for b in n2.buckets:
			if not b.linked or b.otherBucket not in n1.buckets:
				buckets.append(b)

		n = Node(t, Buckets=buckets)

		# The order matters here: we have to remove the old nodes before
		# adding the new one to make sure that the correct buckets end up
		# in the network.
		self.removeNode(n1)
		self.removeNode(n2)
		self.addNode(n)

		return n

	def mergeLinks(self, n, accuracy=1e-4):
		for n1 in n.connectedNodes:
			links = n1.linksConnecting(n)
			buckets1 = [l.bucket1 for l in links if l.bucket1.node is n]
			buckets2 = [l.bucket2 for l in links if l.bucket2.node is n]
			buckets = buckets1 + buckets2
			buckets1 = [l.bucket1 for l in links if l.bucket1.node is n1]
			buckets2 = [l.bucket2 for l in links if l.bucket2.node is n1]
			otherBuckets = buckets1 + buckets2
			b = n.mergeBuckets(buckets)
			b1 = n1.mergeBuckets(otherBuckets)
			l = Link(b, b1)
			compressLink(l, accuracy)

	def internalConnected(self, node):
		return self.nodes.intersection(set(node.connectedNodes))


