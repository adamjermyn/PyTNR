from networkTree import NetworkTree
from latticeNode import latticeNode
from node import Node
from utils import flatten, multiMod
import numpy as np
from copy import deepcopy

class PeriodicNetworkTree(NetworkTree):
	'''
	A PeriodicNetworkTree is a NetworkTree which represents the renormalisation
	of a periodic tensor network.
	'''
	def __init__(self, latticeLength, dimensions, bondArrays, footprints):
		self._latticeLength = latticeLength
		self._dimensions = dimensions
		self._bondArrays = bondArrays
		self._footprints = footprints

		NetworkTree.__init__(self)

		self._siteIndices = np.meshgrid((range(d) for d in dimensions))
		self._sites = np.empty(shape=dimensions, dtype='object')

		self._bonds = [np.empty(shape=dimensions, dtype='object') for _ in bondArrays]

		self._periodicLinks = [set() for _ in dimensions]
		self._allPeriodicLinks = set()

		for si in self._siteIndices:
			self._sites[si] = latticeNode(latticeLength, self)
			for i in range(len(bondArrays)):
				self._bonds[i][si] = self.addNodeFromArray(bondArrays[i])

		for si in self._siteIndices:
			for i in range(len(bondArrays)):
				for q,dj in enumerate(footprints[i]):
					x, dims = multiMod(np.array(si) + np.array(dj))
					l = self._sites[si].addLink(self._bonds[i][x],q)
					if len(dims) > 0:
						self._allPeriodicLinks.add(l)
						l.setPeriodic()
					for i in dims:
						self._periodicLinks[i].add(l)

	def merge(self, mergeL=True, compressL=True, eps=1e-4):
		'''
		Performs the next best merger (contraction) between Nodes based on entropy heuristics.
		The Nodes must be linked to one another. Mergers are disallowed across periodic bonds.

		This method takes three keyword arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''

		while link.periodic():
			try:
				link = self._sortedLinks.pop()
			except KeyError:
				return 0

		assert link in self._topLevelLinks

		n1 = link.bucket1().topNode()
		n2 = link.bucket2().topNode()

		assert n1 in self._topLevelNodes
		assert n2 in self._topLevelNodes
		assert n1 != n2

		n1.merge(n2, mergeL=mergeL, compressL=compressL, eps=eps)
		return 1

	def contract(self, mergeL=True, compressL=True, eps=1e-4):
		'''
		This method contracts the Network to a minimal representation while maintaining
		bonds in the periodic direction.

		This method takes three keywork arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''
		self.trace()

		counter = 0
		done = False

		while not done:
			ret = self.merge(mergeL=mergeL, compressL=compressL, eps=eps)

			if ret == 0:
				done = True

			if counter%20 == 0:
				t = self.largestTopLevelTensor()
				print(len(self.topLevelNodes()),self.topLevelSize(), t.tensor().shape())
			counter += 1


	def expand(self, dimension):
		self._dimensions[dimension] *= 2

		newNodes = set()
		newNodeOldID = {}
		oldNodeNewID = {}
		newBucketOldIDind = {}
		oldBucketNewIDind = {}

		nn = NetworkTree()

		subset = set(self._topLevelNodes)

		# Copy top-level Nodes

		for n in subset:
			m = Node(n.tensor(), self, children=n.children(), logScalar = n.logScalar())
			newNodes.add(m)

			newNodeOldID[n.id()] = m
			oldNodeNewID[m.id()] = n

			for i in range(len(n.buckets())):
				newBucketOldIDind[(n.id(),i)] = n.buckets()[i]
				oldBucketNewIDind[(m.id(),i)] = m.buckets()[i]

		# Link new Nodes

		for oldN in subset:
			newN = newNodeOldID[oldN.id()]

			for ind0, b in enumerate(oldN.buckets()):
				if b.linked():
					otherB = b.otherBucket()

					intersection = set(otherB.nodes()).intersection(subset)

					if len(intersection) > 0:
						assert len(intersection) == 1
						oldNlinked = intersection.pop()
						ind1 = oldNlinked.buckets().index(otherB)

						newNlinked = newNodeOldID[oldNlinked.id()]

						if not newNlinked.buckets()[ind1].linked():
							newN.addLink(newNlinked, ind0, ind1)

		# Fix periodic Links
		# Do this by bucket swapping


		other = 