from collections import defaultdict
import networkx
import numpy as np

from TNRG.Network.cycle import cycle

class cycleBasis:
	def __init__(self, network):
		'''
		A cycleBasis is an object which maintains a cycle basis for a network.
		It also provides helper methods for manipulating cycles while maintaining the basis.

		To construct a cycle basis it suffices to provide a network object.
		'''
		self.network = network

		# Construct graph and cycle basis
		g = network.toGraph()
		cyclesNodes = networkx.cycle_basis(g)
		weights = [np.sum(np.log([n.tensor.size for n in cycle])) for cycle in cyclesNodes]

		# Internally we store cycles as lists of edges
		self._cycles = [[c[i].findLink(c[i-1]) for i in range(len(c))] for c in cyclesNodes]
		self._cycles = [cycle(c, self) for c in self._cycles]

		self.edgeDict = defaultdict(list)

		for c in self.cycles:
			for e in c:
				self.edgeDict[e].append(c)

	@property
	def cycles(self):
		return self._cycles

	def smallest(self):
		if len(self.cycles) == 0:
			return None
		return min(self.cycles, key=lambda x: len(x))

	def hardest(self):
		if len(self.cycles) == 0:
			return None
		return max(self.cycles, key=lambda x: np.sum(np.log([n.tensor.size for n in x.nodes])))

	def freeNodes(self, cycle):
		'''
		Each node in a cycle has two indices connecting to edges in the cycle and a third index
		which is either unlinked or leads to an edge out of the cycle. This method identifies the
		nodes whose third index is involved in no other cycles.
		'''

		nodes = cycle.nodes
		freeNodes = set()

		for n in nodes:
			for b in n.buckets:
				if not b.linked or len(self.edgeDict[b.link]) == 0:
					freeNodes.add(n)

		return freeNodes

	def commonNodes(self, cycle):
		'''
		Each node in a cycle has two indices connecting to edges in the cycle and a third index
		which leads out of the cycle. This method identifies the node pairs such that the third
		index of each node in the pair is involved in another cycle.

		# There is something wrong either with this method or with the edgeDict which prevents us
		# from getting pairs even when there mathematically must be pairs.
		'''
		nodes = cycle.nodes

		pairs = []

		for e in cycle:
			ed = self.edgeDict[e]
			for c in ed:
				if c != cycle:
					# Now we have a cycle that intersects
					inter = []
					for en in c:
						if en in cycle:
							n1 = en.bucket1.node
							n2 = en.bucket2.node
							if cycle.outBucket(n1).linked and cycle.outBucket(n1).link in c:
								inter.append(n1)
							if cycle.outBucket(n2).linked and cycle.outBucket(n2).link in c:
								inter.append(n2)

					for n in inter:
						for m in inter:
							if n != m:
								ind1 = nodes.index(n)
								ind2 = nodes.index(m)
								dist = abs(ind2 - ind1)
								dist = min(dist, len(cycle) - dist)
								pairs.append((n,m,dist))

		return pairs

	def mergeEdge(self, edge):
		'''
		This method merges the nodes on either side of an edge and handles updating the cycles accordingly.
		'''
		n1 = edge.bucket1.node
		n2 = edge.bucket2.node

		assert n1 in self.network.nodes
		assert n2 in self.network.nodes

		links = n1.linksConnecting(n2)

		self.network.mergeNodes(n1, n2)

		for e in links:
			for c in self.edgeDict[e]:
				c.remove(e)
				assert c.valid == False
			del self.edgeDict[e]


	def mergeSmall(self, cycle):
		# This method merges a cycle of size <= 3 and handles updating the cycle accordingly.

		assert len(cycle) <= 3

		while len(cycle) > 0:
			self.mergeEdge(cycle.edges[0])
			cycle.validate()


	def swap(self, edge, b1, b2):
		'''
		This method merges the nodes on either side of the given edge and then
		splits them in such a way that buckets b1 and b2 (which must be buckets of
		these nodes) are on the same node.
		'''

		self.consistencyCheck()
		n1 = edge.bucket1.node
		n2 = edge.bucket2.node

		if b1 not in n1.buckets:
			n1, n2 = n2, n1

		# Catalog chanages
		outBucket1 = set(n1.buckets)
		outBucket1.discard(b1)
		outBucket1.discard(edge.bucket1)
		outBucket1.discard(edge.bucket2)
		outBucket1 = outBucket1.pop()
		outBucket2 = set(n2.buckets)
		outBucket2.discard(b2)
		outBucket2.discard(edge.bucket1)
		outBucket2.discard(edge.bucket2)
		outBucket2 = outBucket2.pop()

		affectedCycles = list(self.edgeDict[edge])
		if outBucket1.linked:
			outLink1 = outBucket1.link
			affectedCycles = affectedCycles + self.edgeDict[outBucket1.link]
		else:
			outLink1 = None
		if outBucket2.linked:
			outLink2 = outBucket2.link
			affectedCycles = affectedCycles + self.edgeDict[outBucket2.link]
		else:
			outLink2 = None

		if b1.linked:
			bLink1 = b1.link
		else:
			bLink1 = None

		if b2.linked:
			bLink2 = b2.link
		else:
			bLink2 = None

		new = []
		for x in affectedCycles:
			if x not in new:
				new.append(x)
		affectedCycles = new

		assert b1 in n1.buckets
		assert b2 in n2.buckets
		assert b1 != edge.bucket1 and b1 != edge.bucket2
		assert b2 != edge.bucket1 and b2 != edge.bucket2

		for c in affectedCycles:
			if len(c) > 0:
				c.fixOrder()

		for c in affectedCycles:
			print("HADIASIHDKASJDAKSD")
			for e in c:
				print(e)
				print(e.bucket1.node)
				print(e.bucket2.node)

		# Perform swap
		n = self.network.mergeNodes(n1, n2)
		nodes = self.network.splitNode(n, ignore=[n.bucketIndex(b1),n.bucketIndex(b2)])
		edgeNew = nodes[0].findLink(nodes[1])

		# Update cycle references
		for c in affectedCycles:
			if (outLink1 in c) != (bLink2 in c):
				print('Antisym',flush=True)
				if edge in c:
					print('with edge',flush=True)
					c.remove(edge)
				else:
					print('without edge',flush=True)
					c.add(edgeNew)
			elif edge in c:
				print('Sym with edge',flush=True)
				ind = c.index(edge)
				c.add(edgeNew)
				c.remove(edge)
			else:
				print('Sym without edge so nothing to do... right?',flush=True)
			c.validate()

		assert len(self.edgeDict[edge]) == 0
		del self.edgeDict[edge]

		self.consistencyCheck()

	def swapCycle(self, cycle, edge):
		'''
		This method merges the nodes on either side of edge and then splits them
		in such a way as to swap the two indices which lead outside of the specified cycle.
		'''

		assert edge in cycle

		n1 = edge.bucket1.node
		n2 = edge.bucket2.node

		assert len(n1.buckets) == 3
		assert len(n2.buckets) == 3

		nodes = cycle.nodes
		cycleBucket1 = cycle.cycleBucket(n1, avoid=edge)
		outBucket2 = cycle.outBucket(n2)

		# This choice of buckets to ignore ensures that the first node in the returned nodes
		# list is the one in the same place in the cycle as n1 but with the outgoing bucket
		# from n2.
		b1 = cycleBucket1
		b2 = outBucket2

		self.swap(edge, b1, b2)

	def walk(self, cycle, n1, n2):
		'''
		This method uses a swap operation to move the out-of-cycle bucket on one
		of n1 or n2 towards the out-of-cycle bucket on the other.
		'''
		nodes = cycle.nodes

		assert n1 in nodes
		assert n2 in nodes

		b1 = cycle.outBucket(n1)
		b2 = cycle.outBucket(n2)

		edge = cycle.nearEdge(n1, n2)
		self.swapCycle(cycle, edge)

		nodes = cycle.nodes
		n1 = None
		n2 = None
		for i,n in enumerate(nodes):
			if cycle.outBucket(n) == b1:
				n1 = n
			elif cycle.outBucket(n) == b2:
				n2 = n

		return n1, n2

	def makeAdjacent(self, cycle, n1, n2):
		'''
		This method takes a cycle containing nodes n1 and n2 and performs
		swaps to put the out-of-cycle buckets of these nodes on adjacent nodes.
		'''

		nodes = cycle.nodes

		assert n1 in nodes
		assert n2 in nodes

		dist = cycle.dist(n1, n2)

		while dist > 1:
			n1, n2 = self.walk(cycle, n1, n2)
			dist = cycle.dist(n1, n2)

		edge = n1.findLink(n2)

		return edge

	def pinch(self, cycle, n1, n2):
		'''
		This method takes a cycle containing adjacent nodes n1 and n2.
		It merges n1 and n2 and then splits them so as to keep the buckets
		which link to the given cycle in a single node. This is just a wrapper
		around swap for the special case described.
		'''
		nodes = cycle.nodes

		edge = n1.findLink(n2)

		assert n1 in nodes
		assert n2 in nodes
		assert cycle.dist(n1, n2) == 1

		cycleBucket1 = cycle.cycleBucket(n1, avoid=edge)
		cycleBucket2 = cycle.cycleBucket(n2, avoid=edge)

		b1 = cycleBucket1
		b2 = cycleBucket2

		print('pinch',len(cycle))
		self.swap(edge, b1, b2)


	def consistencyCheck(self):
		print('Consistency check...')
		for cycle in self.cycles:
			for i in range(len(cycle)):
				e1 = cycle[i]
				e2 = cycle[i-1]
				assert e1.bucket1.node in self.network.nodes
				assert e1.bucket2.node in self.network.nodes
				assert len(set([e1.bucket1.node,e1.bucket2.node]).intersection(set([e2.bucket1.node,e2.bucket2.node]))) > 0
			assert len(set(cycle.nodes)) == len(cycle)

		for cycle in self.cycles:
			for e in cycle:
				cd = self.edgeDict[e]
				assert cycle in cd
				assert len(cd) == len(set(cd))

		print('Consistent!')













