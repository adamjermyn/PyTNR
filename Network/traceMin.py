from collections import defaultdict
import networkx
import numpy as np
import operator
import community
from collections import Counter
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

def hortonGraph(adj, s):
	'''
	adj - Adjacency matrix for the original graph
	s - Set of edges which will cross over between the two copies of the original graph

	Edges are stored as tuples of indices.
	'''
	adjH = np.zeros((2*len(adj),2*len(adj)))
	adjH[:len(adj),:len(adj)] = adj
	adjH[len(adj):,len(adj):] = adj

	for e in s:
		i, j = e
		ip = i + len(adj)
		jp = j + len(adj)

		adjH[ip, j] = adj[i, j]
		adjH[j, ip] = adj[j, i]
		adjH[jp, i] = adj[j, i]
		adjH[i, jp] = adj[i, j]

		adjH[i,j] = 0
		adjH[j,i] = 0

		adjH[ip, jp] = 0
		adjH[jp, ip] = 0

	return adjH

def shortestPaths(sp, n):
	# Returns the shortest path from j -> j+n over all j.
	# Graph must be input as an adjacency matrix.
	dists, predecessors = shortest_path(sp, method='D', directed=False, return_predecessors=True, unweighted=False, overwrite=False)
	bestInd = np.argmin(np.diagonal(dists, offset=n))

	path = []
	j = bestInd
	k = bestInd + n
	while k != j:
		path.append(k)
		k = predecessors[j, k]

	return path

def shortestPathsAlt(g, n):
	# Returns the shortest path from j -> j+n over all j.
	# Graph must be input as a NetworkX graph.
	best = [1e100, None]
	for j in range(n):
		length = 1e100
		length, path = networkx.single_source_dijkstra(g, j, target=j+n, weight='weight', cutoff=best[0])
		if j+n in path.keys():
			length = length[j+n]
			path = path[j+n]
			if length < best[0]:
				best = [length, path]
	path = best[1]
	return path

def minimalCycleBasis(adj):
	n = len(adj)
	m = int(np.sum(adj > 0) / 2)
	k = networkx.number_connected_components(networkx.from_numpy_matrix(adj))

	N = m - n + k

	edges = np.transpose(np.array(np.where(adj > 0)))
	edges = [(min(edges[i]), max(edges[i])) for i in range(len(edges))]

	s = []
	for i in range(N):
		s.append(set([edges[i]]))

	np.set_printoptions(linewidth=150)

	cycles = []
	for i in range(N):
		ghAdj = hortonGraph(adj, s[i])
		path = shortestPaths(csr_matrix(ghAdj), n)

		pathEdges = [(path[i-1] % n, path[i] % n) for i in range(len(path))]
		pathEdges = [(min(e), max(e)) for e in pathEdges]

		ce = Counter(pathEdges)

		pathEdges = [pe for pe in pathEdges if ce[pe] % 2 == 1]

		cycles.append(pathEdges)

		for j in range(i+1, N):
			if len(s[j].intersection(pathEdges)) % 2 == 1:
				s[j].symmetric_difference(s[i])

	return cycles


def prune(adj):
	'''
	This method eliminates all branches from the graph which are not involved in a cycle.
	This is done by constructing the cycle basis and using only those nodes.
	'''
	g = networkx.from_numpy_matrix(adj)
	cycles = networkx.cycles.cycle_basis(g)

	adjNew = np.zeros(adj.shape)

	for c in cycles:
		for i in range(len(c)):
			adjNew[c[i-1], c[i]] = adj[c[i-1], c[i]]
			adjNew[c[i], c[i-1]] = adj[c[i], c[i-1]]

	return networkx.from_numpy_matrix(adjNew)


def pruneGraph(g):
	cycles = networkx.cycles.cycle_basis(g)

	edges = []
	for c in cycles:
		for i in range(len(c)):
			edges.append((c[i-1],c[i]))

	nodes = set()
	for c in cycles:
		nodes.update(c)

	gNew = g.subgraph(nodes)

	for e in gNew.edges():
		if (e[0], e[1]) not in edges and (e[1], e[0]) not in edges:
			gNew.remove_edge(e[0], e[1])

	return gNew


def util(adj):
	u = 0

	g = prune(np.copy(adj))

	components = networkx.connected_components(g)
	components = [g.subgraph(c) for c in components]

	for g in components:

		adj = networkx.adjacency_matrix(g, weight='weight').todense()

		cycles = minimalCycleBasis(adj)

		for c in cycles:
			un = 0
			for i in range(len(c)):
				un += adj[c[i][0],c[i][1]]
			u += un**0.25

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
		print('LEN:::',len(self.g.nodes()))
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

		return (utilNew - self.util)/(np.log(b1.node.tensor.size*b2.node.tensor.size))**2

	def bestSwap(self):
		print('Evaluating...')

		gNew = pruneGraph(self.g)

		best = [1, None, None, None]
		for e in gNew.edges():
			n1 = e[0]
			n2 = e[1]
			l = n1.findLink(n2)
			buckets = set(n1.buckets)
			buckets.discard(l.bucket1)
			buckets.discard(l.bucket2)
			b1 = buckets.pop()
			for b2 in n2.buckets:
				if not b2.linked or b2.otherNode != n1:
					benefit = self.swapBenefit(l, b1, b2)
					if benefit < best[0]:
						best = [benefit, l, b1, b2]

		print('Done.')
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


