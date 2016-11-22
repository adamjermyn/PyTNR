import itertools
from bucket import Bucket

class Node:
	newid = itertools.count().__next__

	def __init__(self, tensor, Buckets=None):
		self.tensor = tensor
		self.id = Node.newid()
		self.network = None

		if Buckets is None:
			Buckets = []

		self.buckets = Buckets

		for b in Buckets:
			b.node = self

	def __str__(self):
		s = 'Node with ID ' + str(self.id) + ' and tensor shape ' + str(self.tensor.shape)
		s = s + '\n'
		for n in self.connectedNodes:
			s = s + str(n.id) + '\n'
		return s

	@property
	def linkedBuckets(self):
		return [b for b in self.buckets if b.linked]

	@property
	def connectedNodes(self):
		return [b.otherBucket.node for b in self.linkedBuckets]

	def findLinks(self, other):
		links = []
		for b in self.linkedBuckets:
			if other == b.otherNode:
				links.append(b.link)
		return links


	def findLink(self, other):
		links = self.findLinks(other)
		if len(links) > 0:
			return links[0]
		else:
			return None

	def linksConnecting(self, other):
		links = []
		for b in self.linkedBuckets:
			if other == b.otherNode:
				links.append(b.link)
		return links

	def indexConnecting(self, other):
		for b in self.linkedBuckets:
			if other == b.otherNode:
				return b.index
		return None

	def bucketIndex(self, b):
		return self.buckets.index(b)

	def mergeBuckets(self, buckets):
		'''
		This method merges the listed buckets.
		In the case of an ArrayTensor this just flattens the tensor along the corresponding axes.
		In the case of a TreeTensor this merges the external legs.

		If the optional network argument is provided, the merged buckets are deregistered
		from it and the new bucket is added to it.
		'''
		inds = [b.index for b in buckets]
		self.tensor = self.tensor.flatten(inds)
		self.buckets = [b for i,b in enumerate(self.buckets) if i not in inds]
		self.buckets.append(Bucket())
		self.buckets[-1].node = self

		network = self.network
		if network is not None:
			network.buckets = network.buckets.difference(set(buckets))
			network.buckets.add(self.buckets[-1])

			if len(network.internalBuckets.intersection(set(buckets))) == len(buckets):
				network.internalBuckets = network.internalBuckets.difference(set(buckets))
				network.internalBuckets.add(self.buckets[-1])
			elif len(network.externalBuckets.intersection(set(buckets))) == len(buckets):
				network.externalBuckets = network.externalBuckets.difference(set(buckets))
				network.externalBuckets.add(self.buckets[-1])
			else:
				raise ValueError('Error: Provided buckets are a mixture of internal and external buckets!')

		return self.buckets[-1]
