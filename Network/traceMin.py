from collections import defaultdict
import networkx
import numpy as np
import operator
from scipy.linalg import expm
from scipy.linalg import eigvals

def util(adj):
	g = networkx.from_numpy_matrix(adj)
#	g = g.to_directed()
#	cycles = networkx.cycles.simple_cycles(g)

	cycles = networkx.cycles.cycle_basis(g)

	u = 0
	for c in cycles:
		for i in range(len(c)):
			u += adj[c[i-1],c[i]]

	return u

class traceMin:
	def __init__(self, network):
		'''
		A traceMin is an object which maintains the adjacency matrix for a network and
		provides helper methods for evaluating functions of this method and evaluating the
		effect of changes to this network.
		'''
		self.network = network

		# Construct graph and cycle basis
		self.refresh()

	def refresh(self):
		self.g = self.network.toGraph()
		self.adj = networkx.adjacency_matrix(self.g, weight='weight').todense()
		self.util = util(self.adj)

	def pretendSwap(self, edge, b1, b2):
		'''
		This method produces the adjacency matrix which would result from the proposed swap.
		Note that it assumes that the link weights remain unchanged, the links just move.
		'''
		adjNew = np.copy(self.adj)

		n1 = b1.node
		n2 = b2.node

		bConn1 = n1.findLink(n2).bucket1
		bConn2 = n1.findLink(n2).bucket2

		if bConn1 not in n1.buckets:
			bConn1, bConn2 = bConn2, bConn1

		ind1 = self.g.nodes().index(n1)
		ind2 = self.g.nodes().index(n2)

		b3 = [b for b in n1.buckets if b != b1 and b != bConn1][0]

		# Nothing happens to links with b1 because stays put

		if b2.linked:
			ind = self.g.nodes().index(b2.otherNode)
			adjNew[ind1, ind] = self.adj[ind2, ind]
			adjNew[ind, ind1] = self.adj[ind, ind2]
			adjNew[ind2, ind] = 0
			adjNew[ind, ind2] = 0

		if b3.linked:
			ind = self.g.nodes().index(b3.otherNode)
			adjNew[ind2, ind] = self.adj[ind1, ind]
			adjNew[ind, ind2] = self.adj[ind, ind1]
			adjNew[ind1, ind] = 0
			adjNew[ind, ind1] = 0

		return adjNew

	def swapBenefit(self, edge, b1, b2):
		'''
		This method identifies how beneficial a given swap is.
		'''

		adjNew = self.pretendSwap(edge, b1, b2)
		utilNew = util(adjNew)

		return (utilNew - self.util)/(np.log(b1.node.tensor.size*b2.node.tensor.size))

	def bestSwap(self):
#		print(self.network)
		best = [1, None, None, None]
		for n1 in self.network.nodes:
			for n2 in n1.connectedNodes:
				edge = n1.findLink(n2)
				for b1 in n1.buckets:
					if not b1.linked or b1.otherNode != n2:
						for b2 in n2.buckets:
							if not b2.linked or b2.otherNode != n1:
								benefit = self.swapBenefit(edge, b1, b2)
								if benefit < best[0]:
									best = [benefit, edge, b1, b2]
		return best

	def mergeEdge(self, edge):
		'''
		This method merges the nodes on either side of an edge and handles updating the cycles accordingly.
		'''
		n1 = edge.bucket1.node
		n2 = edge.bucket2.node

		assert n1 in self.network.nodes
		assert n2 in self.network.nodes

		self.network.mergeNodes(n1, n2)
		self.refresh()

	def mergeSmall(self):
		'''
		This method identifies all mergers which can be performed without increasing rank beyond 3
		and does them.
		'''
		done = set()
		merged = False

		while len(self.network.nodes.intersection(done)) < len(self.network.nodes):
			n1 = next(iter(self.network.nodes.difference(done)))
			for n2 in n1.connectedNodes:
				if len(n2.buckets) < 3 or len(n1.connectedNodes.intersection(n2.connectedNodes)) > 0 or len(n1.findLinks(n2)) > 1:
					self.network.mergeNodes(n1, n2)
					merged = True
					break
			else:
				done.add(n1)

		self.refresh()
		return merged

	def swap(self, edge, b1, b2):
		'''
		This method merges the nodes on either side of the given edge and then
		splits them in such a way that buckets b1 and b2 (which must be buckets of
		these nodes) are on the same node.
		'''
		n1 = edge.bucket1.node
		n2 = edge.bucket2.node

		if b1 not in n1.buckets:
			n1, n2 = n2, n1

		# Perform swap
		n = self.network.mergeNodes(n1, n2)
		nodes = self.network.splitNode(n, ignore=[n.bucketIndex(b1),n.bucketIndex(b2)])

		self.refresh()


