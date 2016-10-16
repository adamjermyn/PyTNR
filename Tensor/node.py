class Node:

	def __init__(self, tensor, network, Buckets=None):
		self.tensor = tensor
		self.network = network
		self.ID = self.network.idCounter

		if Buckets is None:
			Buckets = []

		self.buckets = Buckets

		for b in Buckets:
			b.node = self

		self.network.registerNode(self)

	def __str__(self):
		return 'Node with ID: ' + str(self.ID) + '  and tensor shape ' + str(self.tensor.shape)

	def connected(self):
		c = []
		for b in self.buckets:
			if b.linked():
				c.extend(b.otherNodes())
		return c

	def findLink(self, other):
		for b in self.buckets:
			if b.linked():
				if other in b.otherNodes():
					return b.link()
		return None

	def linksConnecting(self, other):
		links = []
		for b in self.buckets:
			if b.linked():
				if other in b.otherNodes():
					links.append(b.link())
		return links

	def indexConnecting(self, other):
		for i,b in enumerate(self.buckets):
			if b.linked():
				if other in b.otherNodes():
					return i
		return None		

	def bucketIndex(self, b):
		return self.buckets.index(b)
