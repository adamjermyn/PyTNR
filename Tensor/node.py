import itertools

class Node:
	newid = itertools.count().__next__

	def __init__(self, tensor, Buckets=None):
		self.tensor = tensor
		self.id = Node.newid()

		if Buckets is None:
			Buckets = []

		self.buckets = Buckets

		for b in Buckets:
			b.node = self

	def __str__(self):
		return 'Node with ID ' + str(self.id) + ' and tensor shape ' + str(self.tensor.shape)

	@property
	def linkedBuckets(self):
		return [b for b in self.buckets if b.linked]

	@property
	def connectedNodes(self):
		return [b.otherBucket.node for b in self.linkedBuckets]

	def findLink(self, other):
		for b in self.linkedBuckets:
			if other == b.otherNode:
				return b.link
		return None

	def linksConnecting(self, other):
		links = []
		for b in self.linkedBuckets:
			if other == b.otherNode:
				links.append(b.link)
		return links

	def indexConnecting(self, other):
		for i,b in enumerate(self.linkedBuckets):
			if other == b.otherBucket.node:
				return i
		return None		

	def bucketIndex(self, b):
		return self.buckets.index(b)