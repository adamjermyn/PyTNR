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
		self._cycles = [cycle(c) for c in self._cycles]

		self.edgeDict = defaultdict(list)

		for c in self.cycles:
			for e in c:
				self.edgeDict[e].append(c)

	@property
	def cycles(self):
		return list(filter(lambda x: len(x) > 0, self._cycles))

	def smallest(self):
		return min(self.cycles, key=lambda x: len(x))

	def hardest(self):
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
		'''
		nodes = cycle.nodes

		for n in nodes:
			print(n)

		pairs = []

		outEdges = []
		for n in nodes:
			outLinks = list(filter(lambda x: x.otherNode not in cycle, n.linkedBuckets))
			if len(outLinks) > 0:
				outLink = outLinks.pop().link
				outEdges.append((n, outLink))

		for n,e in outEdges:
			print(e)
			print(self.edgeDict[e])

		common = []
		for i in range(len(outEdges)):
			for j in range(i):
				n, en = outEdges[i]
				m, em = outEdges[j]
				cn = self.edgeDict[en]
				cm = self.edgeDict[em]
				inter = []
				for c1 in cn:
					for c2 in cm:
						if c11 == c22 and c11 != ccc:
							inter.append(c1)
				if len(inter) > 0:
					dist = nodes.index(n) - nodes.index(m)
					dist = abs(dist)
					dist = min(dist, len(nodes) - dist)
					common.append((n,m,dist,inter))

		return common

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
			del self.edgeDict[e]


	def mergeSmall(self, cycle):
		# This method merges a cycle of size <= 3 and handles updating the cycle accordingly.

		assert len(cycle) <= 3

		while len(cycle) > 0:
			self.mergeEdge(cycle.edges[0])

	def swap(self, edge, b1, b2):
		'''
		This method merges the nodes on either side of the given edge and then
		splits them in such a way that buckets b1 and b2 (which must be buckets of
		these nodes) are on the same node.
		'''

#		print('Swapping... on edge',edge)
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

		# Perform swap
		n = self.network.mergeNodes(n1, n2)
		nodes = self.network.splitNode(n, ignore=[n.bucketIndex(b1),n.bucketIndex(b2)])
		edgeNew = nodes[0].findLink(nodes[1])

		# Update cycle references
		self.edgeDict[edgeNew] = list(self.edgeDict[edge])
		del self.edgeDict[edge]

		for c in affectedCycles:
			if (outLink1 in c) != (bLink2 in c):
				if edge in c:
					c.remove(edge)
					self.edgeDict[edgeNew].remove(c)
				else:
					self.edgeDict[edgeNew].append(c)

					# Look for the place where the swap broke the cycle
					ind = c.checkConsistency()
					assert ind is not None
					c.insert(ind, edgeNew)
			elif edge in c:
				ind = c.index(edge)
				c.insert(ind, edgeNew)
				c.remove(edge)


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
		for b in n2.buckets:
			print(not b.linked)
			if b.linked:
				print('    ',b.otherNode not in nodes)

		outBucket2 = cycle.outBucket(n2)

		# This choice of buckets to ignore ensures that the first node in the returned nodes
		# list is the one in the same place in the cycle as n1 but with the outgoing bucket
		# from n2.
		b1 = cycleBucket1
		b2 = outBucket2

		self.swap(edge, b1, b2)

	def makeAdjacent(self, cycle, n1, n2):
		'''
		This method takes a cycle containing nodes n1 and n2 and performs
		swaps to put the out-of-cycle buckets of these nodes on adjacent nodes.
		'''

		nodes = cycle.nodes

		assert n1 in nodes
		assert n2 in nodes

		for b in n1.buckets:
			print(not b.linked)
			if b.linked:
				print('   ',b.otherNode not in nodes)

		for b in n2.buckets:
			print(not b.linked)
			if b.linked:
				print('   ',b.otherNode not in nodes)

		for n in nodes:
			print(n)

		print('asdkj',n1)
		print('akdha',n2)
		print(len(self.smallest()))

		outBucket1 = cycle.outBucket(n1)
		outBucket2 = cycle.outBucket(n2)

		ind1 = nodes.index(n1)
		ind2 = nodes.index(n2)

		# Make sure ind1 < ind2
		if ind2 < ind1:
			cycle.reverse()
			nodes = cycle.nodes
			ind1 = nodes.index(n1)
			ind2 = nodes.index(n2)

		# Check to see if the best route goes through the zero position
		# and if so rotate the cycle to avoid this.
		dist = ind2 - ind1
		if dist > len(cycle) - dist:
			cycle.rotate(ind2) # This makes ind2 == 0
			nodes = cycle.nodes
			dist = len(cycle) - dist
			ind1 = nodes.index(n1)
			ind2 = nodes.index(n2)

			# The rotation leaves ind2 < ind1 so we relabel n1 and n2
			n1, n2 = n2, n1
			ind1, ind2 = ind2, ind1
			outBucket1, outBucket2 = outBucket2, outBucket1

		# Now we perform swaps to move outBucket2 towards outBucket1
		assert n1 == nodes[ind1]
		assert n2 == nodes[ind2]

		edge = cycle[ind2 - 1]
		assert n2 == edge.bucket1.node or n2 == edge.bucket2.node

		self.consistencyCheck()

		while ind2 > ind1 + 1:
			self.swapCycle(cycle, edge)
			self.consistencyCheck()
			ind2 -= 1
			edge = cycle[ind2 - 1]

			n1 = edge.bucket1.node
			n2 = edge.bucket2.node

			assert outBucket2 in n1.buckets or outBucket2 in n2.buckets

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
		dist = abs(nodes.index(n1) - nodes.index(n2))
		dist = min(dist, len(nodes) - dist)
		assert dist == 1

		cycleBucket1 = cycle.cycleBucket(n1, avoid=edge)
		cycleBucket2 = cycle.cycleBucket(n2, avoid=edge)

		b1 = cycleBucket1
		b2 = cycleBucket2

		self.swap(edge, b1, b2)


	def consistencyCheck(self):
		print('Consistency check...')
		for cycle in self.cycles:
#			print('ho')
			for i in range(len(cycle)):
				e1 = cycle[i]

#				print(e1)
#				print(e1.bucket1.node)
#				print(e1.bucket2.node)

				e2 = cycle[i-1]
				assert e1.bucket1.node in self.network.nodes
				assert e1.bucket2.node in self.network.nodes
				assert len(set([e1.bucket1.node,e1.bucket2.node]).intersection(set([e2.bucket1.node,e2.bucket2.node]))) > 0
			assert len(set(cycle.nodes)) == len(cycle)
		print('Consistent!')














