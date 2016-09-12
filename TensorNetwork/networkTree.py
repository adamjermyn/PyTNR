from tensor import Tensor
from link import Link
from bucket import Bucket
from node import Node
import numpy as np
from compress import compress
from mergeLinks import mergeLinks
from priorityQueue import PriorityQueue
from network import Network

class NetworkTree:
	'''
	A NetworkTree is an object storing Nodes as well as providing helper methods
	for manipulating those Nodes.
	'''

	def __init__(self):
		'''
		Method for initializing an empty Network.
		'''
		self._nodes = set()
		self._topLevelNodes = set()
		self._bottomLevelNodes = set()
		self._allLinks = set()
		self._topLevelLinks = set()
		self._cutLinks = set()
		self._sortedLinks = PriorityQueue()
		self._idDict = {}
		self._idCounter = 0

	def __str__(self):
		'''
		Returns string representations of all top-level Nodes in the Network.
		The are no guarantees of the order in which this is done.
		'''
		s = 'Network\n'
		for n in self._topLevelNodes:
			s = s + str(n) + '\n'
		return s

	def size(self):
		'''
		Returns the sum of the sizes of all Tensors in the Network.
		'''
		return sum(n.tensor().size() for n in self._nodes)

	def topLevelSize(self):
		'''
		Returns the sum of the sizes of all Tensors belonging to top-level Nodes in the Network.
		'''
		return sum(n.tensor().size() for n in self._topLevelNodes)

	def nodes(self):
		'''
		Returns a copy of the set of all Nodes in the Network.
		'''
		return set(self._nodes)

	def topLevelNodes(self):
		'''
		Returns a copy of the set of all top-level Nodes in the Network.
		'''
		return set(self._topLevelNodes)

	def topLevelLinks(self):
		'''
		Returns a copy of the set of all top-level Links in the Network.
		'''
		return set(self._topLevelLinks)

	def largestTensor(self):
		'''
		Returns the Node containing the largest Tensor in the Network.
		'''
		sizeGetter = lambda n: n.tensor().size()
		return max(self._nodes, key=sizeGetter)

	def largestTopLevelTensor(self):
		'''
		Returns the top-level Node with the largest Tensor in the Network.
		'''
		sizeGetter = lambda n: n.tensor().size()
		return max(self._topLevelNodes, key=sizeGetter)

	def topLevelRepresentation(self):
		'''
		Returns the tensor product of all top-level Tensors along with
		a list of corresponding Buckets in the same order as the indices.
		'''
		arr = np.array([1.])
		logS = 0
		bucketList = []

		for n in self._topLevelNodes:
			print(n)
			arr = np.tensordot(arr, n.tensor().array(), axes=0)
			logS += n.logScalar()
			for b in n.buckets():
				bucketList.append(b)

		return arr[0], logS, bucketList

	def registerLink(self, link):
		'''
		Registers a new Link in the Network.
		This should only be called when a Link is created.
		'''
		assert link not in self._allLinks
		assert link not in self._topLevelLinks

		self._allLinks.add(link)
		self.registerLinkTop(link)

	def deregisterLink(self, link):
		'''
		De-registers a Link from the Network.
		This should only be used when deleting a Link.
		'''
		assert link in self._allLinks
		assert link in self._topLevelLinks or link in self._cutLinks

		self._allLinks.remove(link)
		if link in self._topLevelLinks:
			self._topLevelLinks.remove(link)
		else:
			self._cutLinks.remove(link)
		self._sortedLinks.remove(link)

	def registerLinkTop(self, link):
		'''
		Registers a Link in the Network as being top-level.
		This is called by registerLink, and hence is used when a Link is created.
		It is also called when a Link is deleted (so that the children of that Link
		may become top-level).
		'''
		assert link not in self._topLevelLinks
		assert link.bucket1().topNode() in self._topLevelNodes
		assert link.bucket2().topNode() in self._topLevelNodes

		self._topLevelLinks.add(link)
		self._sortedLinks.add(link, link.mergeEntropy())

	def deregisterLinkTop(self, link):
		'''
		De-registers a Link in the Network from being top-level.
		This should be called only when a Link is compressed, deleted,
		or merged with another Link.
		'''
		assert link in self._allLinks
		assert link in self._topLevelLinks

		self._topLevelLinks.remove(link)
		self._sortedLinks.remove(link)


	def registerLinkCut(self, link):
		'''
		Registers a link as having been cut and de-registers is from the top-level.
		This should only be called when a Link is cut or traced.
		This occurs when, upon compression, the Link is reduced to bond dimension 1
		or when a Link leads from a Tensor to itself.
		'''
		assert link not in self._cutLinks
		assert link in self._topLevelLinks
		assert link in self._allLinks

		self._cutLinks.add(link)
		self.deregisterLinkTop(link)

	def deregisterLinkCut(self, link):
		'''
		De-registers a Link from being cut and adds it to the top-level.
		This is called only when a Node directly above one on either side of
		a cut Link is deleted, as that indicates that the bond ought to be
		active once more (the compression resulting in it being cut has been
		undone).
		'''
		assert link in self._cutLinks
		assert link in self._allLinks
		assert link not in self._topLevelLinks

		self._cutLinks.remove(link)
		self.registerLinkTop(link)

	def updateSortedLinkList(self, link):
		'''
		Updates the position of the given Link in the priority queue of Links
		to be contracted.
		'''
		self._sortedLinks.remove(link)
		self._sortedLinks.add(link, link.mergeEntropy())

	def registerNode(self, node):
		'''
		Registers a new Node in the Network.
		This should only be called when registering a new Node.
		'''
		assert node not in self._nodes
		assert node not in self._topLevelNodes
		assert len(set(node.children()).intersection(self._topLevelNodes)) == len(node.children())

		self._nodes.add(node)
		self._topLevelNodes.add(node)
		if len(node.children()) == 0:
			self._bottomLevelNodes.add(node)

		children = node.children()
		for c in children:
			self._topLevelNodes.remove(c)

		assert len(set(node.children()).intersection(self._topLevelNodes)) == 0


	def deregisterNode(self, node):
		'''
		De-registers a Node from the Network.
		This should only be called when deleting a Node.
		This also handles updating the link registration
		in the event that the Node was formed from contracting
		a Link.
		'''

		self._nodes.remove(node)
		self._topLevelNodes.remove(node)

		children = node.children()

		self._topLevelNodes.update(children)

		for b in node.buckets():
			assert b.topNode() == node

		if len(children) == 2:
			links = children[0].linksConnecting(children[1])
			for l in links:
				self.registerLinkTop(l)

	def nextID(self):
		'''
		Returns the next unused ID number in the Network.
		'''
		idd = self._idCounter
		self._idCounter += 1
		return idd


	def addNodeFromArray(self, arr):
		'''
		Takes as input an array and constructs a Tensor and Node around it,
		then adds the Node to this Network.
		'''
		t = Tensor(arr.shape, arr)
		return Node(t, self, Buckets=[Bucket(self) for _ in range(len(arr.shape))])

	def filterSubset(self, subset):
		'''
		Given a subset of the Nodes in this Network obeying the property that
		none are ancestors of others, this method finds and returns the highest-level
		set of Nodes in the Network which have the following properties:
		-	Any Node which is not a descentend of these Nodes is an ancestor of at least
			one of them.
		-	None of these Nodes are ancestors of any others.
		-	The specified Nodes are all in the returned set.

		This is done by starting with the top-level Nodes.
		At each stage we remove a Node which is not in the subset of interest
		from the working set. We then check if its descendents include any Nodes
		from the subset of interest and if so we add its children to the working
		set.
		'''
		nn = Network(self.topLevelNodes())
		remaining = nn.nodes().difference(subset)

		while len(nn.nodes().intersection(subset)) < len(subset):
			n = remaining.pop()
			if len(set(subset).intersection(n.allNChildren())) > 0:
				# Means we want to move down to this Node's children
				nn.descend(n)
				remaining = nn.nodes().difference(subset)
			elif len(set(subset).intersection(set(n.ancestors()))) > 0:
				# Means we want to move up to this Node's parent
				nn.ascend(n)
				remaining = nn.nodes().difference(subset)

			for n in nn.nodes():
				for b in n.buckets():
					assert len(set(b.otherNodes()).intersection(nn.nodes())) == 1

		return nn

	def copySubset(self, subset):
		'''
		Given a subset of the Nodes in this Network, this method
		produces a Network containing a copy of these Nodes connected
		as they are in this Network. In addition this method returns
		dictionaries mapping back and forth between the Nodes and Buckets
		of the new and old Network.

		todo: flesh out these docs

		'''

		newNodeOldID = {}
		oldNodeNewID = {}
		newBucketOldIDind = {}
		oldBucketNewIDind = {}

		nn = NetworkTree()

		# Copy Nodes
		for n in subset:
			m = nn.addNodeFromArray(n.tensor().array())

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

		return nn, newNodeOldID, oldNodeNewID, newBucketOldIDind, oldBucketNewIDind

	def view(self, nodes, mergeL=True, compressL=True, eps=1e-4):
		'''
		This method calculates the effective Tensor represented by the Network connecting to the given Nodes.

		The way we do this is straightforward: we go through marking Nodes as rejected
		as if we were deleting them. This includes following compression/merger bonds
		where appropriate. We then copy the highest-level Nodes remaining into a new
		Network and contract it.

		This method takes three keywork arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''
		new = self.filterSubset(nodes).nodes()
		assert len(set(nodes).intersection(new)) == len(nodes)
		for n in new:
			for b in n.buckets():
				assert len(set(b.otherNodes()).intersection(new)) == 1
		new = new.difference(nodes)
		assert len(set(nodes).intersection(new)) == 0
		nn, newNodeOldID, oldNodeNewID, newBucketOldIDind, oldBucketNewIDind = self.copySubset(new)

		# Contract new Network
		nn.contract(mergeL=mergeL, compressL=compressL, eps=eps)

		# Build contracted Tensor and bucketList
		t = nn.largestTopLevelTensor()

		arr, logS, buckets = nn.topLevelRepresentation()
		bucketList = []

		for n in nn.topLevelNodes():
			for b in n.buckets():
				nb = b.bottomNode()
				ind = nb.bucketIndex(b)
				oldNode = oldNodeNewID[nb.id()]
				bucketList.append(oldNode.buckets()[ind].otherBucket())

		return nn, arr, bucketList

	def trace(self):
		'''
		Traces over all Nodes in the Network.
		'''
		nodes = list(self.topLevelNodes())
		for n in nodes:
			n.trace()

	def merge(self, mergeL=True, compressL=True, eps=1e-4):
		'''
		Performs the next best merger (contraction) between Nodes based on entropy heuristics.
		The Nodes must be linked to one another.

		This method takes three keyword arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''

		link = self._sortedLinks.pop()

		assert link in self._topLevelLinks

		n1 = link.bucket1().topNode()
		n2 = link.bucket2().topNode()

		assert n1 in self._topLevelNodes
		assert n2 in self._topLevelNodes
		assert n1 != n2

		n1.merge(n2, mergeL=mergeL, compressL=compressL, eps=eps)


	def linkMerge(self, compressL=False, eps=1e-4):
		'''
		This method checks all Nodes for potential Link mergers and performs any it finds.
		This method takes two keyword arguments:
			compressL	-	Attempts to compress all Links (if any) resulting from a Link merger.
			eps			-	The accuracy of the compression to perform.
		'''
		done = set()
		todo = set(self._topLevelNodes)

		while len(todo) > 0:
			n = todo.pop()
			nn, d, new = n.linkMerge(compressL=compressL, eps=eps)

			todo = todo.difference(d)
			todo = todo | new

			done.add(nn)

	def contract(self, mergeL=True, compressL=True, eps=1e-4):
		'''
		This method contracts the Network to a minimal representation.

		This method takes three keywork arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''
		self.trace()

		counter = 0
		while self._sortedLinks.length > 0:
			self.merge(mergeL=mergeL, compressL=compressL, eps=eps)

			if counter%20 == 0:
				t = self.largestTopLevelTensor()
				print(len(self.topLevelNodes()),self.topLevelSize(), t.tensor().shape())
			counter += 1

	def compressLinks(self, eps=1e-4):
		'''
		This method attempts to compress all top-level Links in the Network.

		This method takes one keyword argument:
			eps	-	The accuracy of the compression to perform.
		'''

		compressed = set()

		while len(compressed) < len(self.topLevelLinks()):
			todo = self.topLevelLinks().difference(compressed)
			todo = list(todo)
			link, _, _ = compress(todo[0], eps=eps)
			compressed.add(link)

	def optimize(self, mergeL=True, compressL=True, eps=1e-4):
		todo = set(self._bottomLevelNodes)
		done = set()

		while len(todo) > 0:
			print(len(todo), len(done))
			n = todo.pop()

			canDo = True
			for c in n.children():
				if c not in done:
					canDo = False
					todo.add(c)
			if not canDo:
				# SHould add back to todo
				continue
			else:
				optimized = False
				for b in n.buckets():
					if b.linked():
						link = b.link()
						numC = len(link.children())
						if b.numNodes() == 1 and numC == 1:
							# Means that the Node was generated
							# by compressing this Link.
							# Note that there can be at most one such Link
							# for any Node, so we don't mind if the loop
							# continues after we compress.
							if link.compressed() and not link.optimized():
								n1 = n
								n2 = b.otherNodes()[0]
								assert n2 in b.otherBucket().nodes()

								n11 = n1.children()[0]
								n22 = n2.children()[0]
								assert b in n1.buckets()
								assert n1 in b.nodes()
								assert b.otherBucket() in n2.buckets()
								assert n2 in b.otherBucket().nodes()
								assert n11 in n22.connected()

								done.add(n11)
								done.add(n22)

								_, arr, bs = self.view([n11, n22], mergeL=mergeL, compressL=compressL, eps=eps)
								print('Optimizing...',len(arr.shape),len(bs))

								# Means we just compressed a single Link
								prevLink = link.children()[0]

								n1.delete() # We only need to delete one of them, as this deletes the other.


								newLink, n1, n2 = compress(prevLink, optimizerArray=arr, optimizerBuckets=bs, eps=eps)

								self.contract(mergeL=mergeL, compressL=compressL, eps=eps)

								done.add(n1)
								done.add(n2)

								if n1.parent() is not None:
									todo.add(n1.parent())
								if n2.parent() is not None:
									todo.add(n2.parent())

								optimized = True
								break
				if not optimized and n.parent() is not None:
					done.add(n)
					todo.add(n.parent())

