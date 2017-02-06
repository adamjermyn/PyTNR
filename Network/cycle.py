import itertools


class cycle:
	newid = itertools.count().__next__

	def __init__(self, edges, basis):
		self.id = cycle.newid()
		self.edges = edges
		self.iterIndex = 0
		self.basis = basis
		self.valid = False
		self.validate()

	def __hash__(self):
		return self.id

	def __len__(self):
		return len(self.edges)

	def __contains__(self, edge):
		return (edge in self.edges)

	def __getitem__(self, key):
		assert self.valid
		return self.edges[key]

	def __iter__(self):
		assert self.valid
		self.iterIndex = 0
		return self

	def __next__(self):
		assert self.valid
		try:
			result = self.edges[self.iterIndex]
		except IndexError:
			raise StopIteration
		self.iterIndex += 1
		return result

	def cycleBucket(self, node, avoid=None, validating=False):
		if not validating:
			assert self.valid
		for b in node.buckets:
			if b.linked and b.link in self.edges and b.link != avoid:
				return b
		raise ValueError('Error: Node contains no bucket linking to this cycle.')

	def outBucket(self, node):
		assert self.valid
		for b in node.buckets:
			if not b.linked or b.link not in self.edges:
				return b
		raise ValueError('Error: Node contains no bucket linking out of this cycle.')

	def fixOrder(self):
		# Ensures that the cycle is ordered
		e = self.edges[0]
		edges = []
		while e not in edges:
			edges.append(e)

			n1 = e.bucket1.node
			n2 = e.bucket2.node

			# Determine the buckets from n1 and n2 to other links in the cycle
			b1 = self.cycleBucket(n1, avoid=e, validating=True)
			b2 = self.cycleBucket(n2, avoid=e, validating=True)

			if b1.link not in edges:
				e = b1.link
			else:
				e = b2.link

		assert len(self.edges) == len(set(edges))
		self.edges = edges

	def validate(self):
		if len(self) == 0:
			print('Removing self with id',self.id)
			self.basis.cycles.remove(self)
		else:
			self.fixOrder() # The process of fixing the order validates the cycle
			self.valid = True

	def index(self, value):
		assert self.valid
		return self.edges.index(value)

	def remove(self, edge):
		self.basis.edgeDict[edge].remove(self)
		self.edges.remove(edge)
		assert edge not in self
		self.valid = False

	def add(self, edge):
		self.basis.edgeDict[edge].append(self)
		self.edges.insert(0, edge)
		self.valid = False

	def reverse(self):
		self.edges = self.edges[::-1]

	def rotate(self, newZero):
		self.edges = self.edges[newZero:] + self.edges[:newZero]

	@property
	def nodes(self):
		assert self.valid
		'''
		This method returns an ordered list of nodes corresponding to the cycle.
		The first node is the common one between edges 0 and -1.
		'''

		nodes = []

		self.fixOrder()

		for i in range(len(self.edges)):
			e1 = self.edges[i-1]
			e2 = self.edges[i]
			common = set([e1.bucket1.node,e1.bucket2.node]).intersection(set([e2.bucket1.node,e2.bucket2.node]))
			assert len(common) > 0
			nodes.extend(common)
		assert len(set(nodes)) == len(self.edges) # No duplicates allowed!

		return nodes

	def dist(self, n1, n2):
		'''
		Returns the shortest distance between the nodes
		'''
		assert self.valid
		nodes = self.nodes
		ind1 = nodes.index(n1)
		ind2 = nodes.index(n2)
		dist = abs(ind2 - ind1)
		dist = min(dist, len(self) - dist)
		return dist

	def nearEdge(self, n1, n2):
		'''
		Returns one of the edges between the two nodes along the shortest path
		'''
		assert self.valid
		nodes = self.nodes
		ind1 = nodes.index(n1)
		ind2 = nodes.index(n2)
		if ind2 < ind1:
			n1, n2 = n2, n1
			ind2, ind1 = ind1, ind2
		dist = ind2 - ind1
		if dist < len(self) - dist:
			return self.edges[ind2 - 1]
		else:
			return self.edges[ind2]

	def checkConsistency(self):
		for i in range(len(self)):
			if len(set([self[i-1].bucket1.node,self[i-1].bucket2.node]).intersection(set([self[i].bucket1.node,self[i].bucket2.node]))) == 0:
				return i
