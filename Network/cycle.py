import itertools


class cycle:
	newid = itertools.count().__next__

	def __init__(self, edges):
		self.id = cycle.newid()
		self.edges = edges
		self.iterIndex = 0

	def __hash__(self):
		return self.id

	def __len__(self):
		return len(self.edges)

	def __contains__(self, edge):
		return (edge in self.edges)

	def __getitem__(self, key):
		return self.edges[key]

	def __setitem__(self, key, value):
		self.edges[key] = value

	def __iter__(self):
		self.iterIndex = 0
		return self

	def __next__(self):
		try:
			result = self.edges[self.iterIndex]
		except IndexError:
			raise StopIteration
		self.iterIndex += 1
		return result

	def index(self, value):
		return self.edges.index(value)

	@property
	def nodes(self):
		'''
		This method returns an ordered list of nodes corresponding to the cycle.
		The first node is the common one between edges 0 and -1.
		'''

		nodes = []

		for i in range(len(self.edges)):
			e1 = self.edges[i-1]
			e2 = self.edges[i]
			nodes.extend(set([e1.bucket1.node,e1.bucket2.node]).intersection(set([e2.bucket1.node,e2.bucket2.node])))

		assert len(set(nodes)) == len(self.edges) # No duplicates allowed!

		return nodes

	def remove(self, edge):
		self.edges.remove(edge)

	def insert(self, index, edge):
		self.edges.insert(index, edge)

	def reverse(self):
		self.edges = self.edges[::-1]

	def rotate(self, newZero):
		self.edges = self.edges[newZero:] + self.edges[:newZero]

	def cycleBucket(self, node, avoid=None):
		for b in node.buckets:
			if b.linked and b.link in self.edges and b.link != avoid:
				return b
		raise ValueError('Error: Node contains no bucket linking to this cycle.')

	def outBucket(self, node):
		for b in node.buckets:
			if not b.linked or b.link not in self.edges:
				return b
		raise ValueError('Error: Node contains no bucket linking out of this cycle.')

	def checkConsistency(self):
		for i in range(len(self)):
			if len(set([self[i-1].bucket1.node,self[i-1].bucket2.node]).intersection(set([self[i].bucket1.node,self[i].bucket2.node]))) == 0:
				return i
