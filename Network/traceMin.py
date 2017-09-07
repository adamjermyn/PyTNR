from collections import defaultdict
import networkx
import numpy as np
import operator
from collections import Counter
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

	g = prune(np.copy(adj))

	u = 0

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
	def __init__(self, network, otherNodes):
		'''
		A traceMin is an object which maintains the adjacency matrix for a network and
		provides helper methods for evaluating functions of this method and evaluating the
		effect of changes to this network.

		otherNodes is a list of other nodes containing tree tensors whose graphs should be
		accomodated in evaluating loops.
		'''
		otherNodes = None #TODODODODODODODODODODO
		if otherNodes is None:
			otherNodes = []

		self.network = network
		self.otherNodes = otherNodes
		self.diffVals = {}

		# Construct graph and cycle basis
		self.refresh()

	def refresh(self):
		graphs = []
		graphs.append(self.network.toGraph())
		for n in self.otherNodes:
			if hasattr(n.tensor,'compressedSize'):
				graphs.append(n.tensor.network.toGraph())

		assert len(graphs) > 0

		u = networkx.Graph(networkx.union_all(graphs))

		for n in self.otherNodes:
			if hasattr(n.tensor,'compressedSize'):
				for b in n.tensor.network.externalBuckets:
					if b.linked:
						u.add_edge(b.node, b.otherNode, weight=np.log(b.node.tensor.size*b.otherNode.tensor.size))

		self.selfGraph = self.network.toGraph()
		self.g = u

		self.adj = networkx.adjacency_matrix(self.g, weight='weight').todense()
		self.util = util(self.adj)


	def pretendSwapGraph(self, g, edge, b1, b2):
		'''
		This method produces the graph which would result from the proposed swap.
		Note that it assumes that the link weights remain unchanged, the links just move.
		'''
		n1 = b1.node
		n2 = b2.node

		bConn1 = n1.findLink(n2).bucket1
		bConn2 = n1.findLink(n2).bucket2

		if bConn1 not in n1.buckets:
			bConn1, bConn2 = bConn2, bConn1

		ind1 = self.g.nodes().index(n1)
		ind2 = self.g.nodes().index(n2)

		b3 = [b for b in n1.buckets if b != b1 and b != bConn1][0]

		# Nothing happnes to links with b1 because it stays put

		if b2.linked:
			n = b2.otherNode
			if n in g:
				g.add_edge(n1, n, weight=g.edge[n1][n2]['weight'])
				g.remove_edge(n, n2)

		if b3.linked:
			n = b3.otherNode
			if n in g:
				g.add_edge(n2, n, weight=g.edge[n1][n2]['weight'])
				g.remove_edge(n, n1)

		return g

	def swapBenefit(self, g, basis, edge, b1, b2):
		'''
		This method identifies how beneficial a given swap is.
		'''

		# First we compute the subgraph corresponding to all cycles
		# which include either of the nodes of interest.
		n1 = b1.node
		n2 = b2.node

		nodes = set()
		for c in basis:
			if n1 in c or n2 in c:
				nodes.update(c)

		logger.debug('Computing swap benefit across ' + str(len(g.nodes())) + ' nodes with respect to basis with ' + str(len(nodes)) + ' nodes and ' + str(len(self.diffVals)) + ' cached values.')

		cacheSet = set(nodes)

		cacheSet = frozenset(cacheSet)
		if cacheSet in self.diffVals:
			logger.debug('Cache hit.')
			return self.diffVals[cacheSet]

		logger.debug('Not cached. Recomputing on',len(nodes),'nodes.')
		subG = g.subgraph(nodes)

		adjCurrent = networkx.adjacency_matrix(subG, weight='weight').todense()
		u = util(adjCurrent)
		subG = self.pretendSwapGraph(subG, edge, b1, b2)
		adjNew = networkx.adjacency_matrix(subG, weight='weight').todense()
		uNew = util(adjNew)

		diff = uNew - u
#		diff *= (n1.tensor.size*n2.tensor.size)

		self.diffVals[cacheSet] = diff

		return diff

	def bestSwap(self):
		print('Evaluating...')

		gNew = pruneGraph(self.g)

		basis = networkx.cycle_basis(gNew)

		best = [1e100, None, None, None]

		for e in gNew.edges():
			if e in self.selfGraph.edges() or (e[1], e[0]) in self.selfGraph.edges():
				n1 = e[0]
				n2 = e[1]
				l = n1.findLink(n2)
				buckets = set(n1.buckets)
				buckets.discard(l.bucket1)
				buckets.discard(l.bucket2)
				b1 = buckets.pop()
				for b2 in n2.buckets:
					if not b2.linked or b2.otherNode != n1:
						benefit = self.swapBenefit(self.g, basis, l, b1, b2)
						if benefit < best[0]:
							best = [benefit, l, b1, b2]
		print('Done.',best,gNew, self.g)
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
		mergedAny = False

		for n in self.network.nodes:
			for n2 in n.connectedNodes:
				assert n2 in n.connectedNodes
				assert n in n2.connectedNodes

		while len(self.network.nodes.intersection(done)) < len(self.network.nodes):
			n1 = next(iter(self.network.nodes.difference(done)))
	
			merged = False
			for n2 in n1.connectedNodes:
				if not merged and (len(n2.buckets) < 3 or len(n1.connectedNodes.intersection(n2.connectedNodes)) > 0 or len(n1.findLinks(n2)) > 1):
					self.network.mergeNodes(n1, n2)
					merged = True
					mergedAny = True

			if not merged:
				done.add(n1)


		self.refresh()

		return mergedAny

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


