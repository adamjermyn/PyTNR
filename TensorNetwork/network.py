from tensor import Tensor
from link import Link
from bucket import Bucket
from node import Node
import numpy as np
from compress import compress
from mergeLinks import mergeLinks
from priorityQueue import PriorityQueue

class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods
	for manipulating those Nodes.
	'''

	def __init__(self):
		'''
		Method for initializing an empty Network.
		'''
		self.__nodes = set()
		self.__topLevelNodes = set()
		self.__bottomLevelNodes = set()
		self.__allLinks = set()
		self.__topLevelLinks = set()
		self.__cutLinks = set()
		self.__sortedLinks = PriorityQueue()
		self.__idDict = {}
		self.__idCounter = 0

	def __str__(self):
		'''
		Returns string representations of all top-level Nodes in the Network.
		The are no guarantees of the order in which this is done.
		'''
		s = 'Network\n'
		for n in self.__topLevelNodes:
			s = s + str(n) + '\n'
		return s

	def size(self):
		'''
		Returns the sum of the sizes of all Tensors in the Network.
		'''
		return sum(n.tensor().size() for n in self.__nodes)

	def topLevelSize(self):
		'''
		Returns the sum of the sizes of all Tensors belonging to top-level Nodes in the Network.
		'''
		return sum(n.tensor().size() for n in self.__topLevelNodes)

	def nodes(self):
		'''
		Returns a copy of the set of all Nodes in the Network.
		'''
		return set(self.__nodes)

	def topLevelNodes(self):
		'''
		Returns a copy of the set of all top-level Nodes in the Network.
		'''
		return set(self.__topLevelNodes)

	def topLevelLinks(self):
		'''
		Returns a copy of the set of all top-level Links in the Network.
		'''
		return set(self.__topLevelLinks)

	def largestTensor(self):
		'''
		Returns the Node containing the largest Tensor in the Network.
		'''
		sizeGetter = lambda n: n.tensor().size()
		return max(self.__nodes, key=sizeGetter)

	def largestTopLevelTensor(self):
		'''
		Returns the top-level Node with the largest Tensor in the Network.
		'''
		sizeGetter = lambda n: n.tensor().size()
		return max(self.__topLevelNodes, key=sizeGetter)

	def registerLink(self, link):
		'''
		Registers a new Link in the Network.
		This should only be called when a Link is created.
		'''
		assert link not in self.__allLinks
		assert link not in self.__topLevelLinks

		self.__allLinks.add(link)
		self.registerLinkTop(link)

	def deregisterLink(self, link):
		'''
		De-registers a Link from the Network.
		This should only be used when deleting a Link.
		'''
		assert link in self.__allLinks
		assert link in self.__topLevelLinks or link in self.__cutLinks

		self.__allLinks.remove(link)
		if link in self.__topLevelLinks:
			self.__topLevelLinks.remove(link)
		else:
			self.__cutLinks.remove(link)
		self.__sortedLinks.remove(link)

	def registerLinkTop(self, link):
		'''
		Registers a Link in the Network as being top-level.
		This is called by registerLink, and hence is used when a Link is created.
		It is also called when a Link is deleted (so that the children of that Link
		may become top-level).
		'''
		assert link not in self.__topLevelLinks
		assert link.bucket1().topNode() in self.__topLevelNodes
		assert link.bucket2().topNode() in self.__topLevelNodes

		self.__topLevelLinks.add(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

	def deregisterLinkTop(self, link):
		'''
		De-registers a Link in the Network from being top-level.
		This should be called only when a Link is traced out,
		compressed, or merged with another Link.
		'''
		assert link in self.__allLinks
		assert link in self.__topLevelLinks

		self.__topLevelLinks.remove(link)
		self.__sortedLinks.remove(link)


	def registerLinkCut(self, link):
		'''
		Registers a link as having been cut and de-registers is from the top-level.
		This should only be called when a Link is cut.
		This occurs when, upon compression, the Link is reduced to bond dimension 1.
		'''
		assert link not in self.__cutLinks
		assert link in self.__topLevelLinks
		assert link in self.__allLinks

		self.__cutLinks.add(link)
		self.deregisterLinkTop(link)

	def deregisterLinkCut(self, link):
		'''
		De-registers a Link from being cut and adds it to the top-level.
		This is called only when a Node directly above one on either side of
		a cut Link is deleted, as that indicates that the bond ought to be
		active once more (the compression resulting in it being cut has been
		undone).
		'''
		assert link in self.__cutLinks
		assert link in self.__allLinks
		assert link not in self.__topLevelLinks

		self.__cutLinks.remove(link)
		self.registerLinkTop(link)

	def updateSortedLinkList(self, link):
		'''
		Updates the position of the given Link in the priority queue of Links
		to be contracted.
		'''
		self.__sortedLinks.remove(link)
		self.__sortedLinks.add(link, link.mergeEntropy())

	def registerNode(self, node):
		'''
		Registers a new Node in the Network.
		This should only be called when registering a new Node.
		'''
		assert node not in self.__nodes
		assert node not in self.__topLevelNodes
		assert len(set(node.children()).intersection(self.__topLevelNodes)) == len(node.children())

		self.__nodes.add(node)
		self.__topLevelNodes.add(node)
		if len(node.children()) == 0:
			self.__bottomLevelNodes.add(node)

		children = node.children()
		for c in children:
			self.__topLevelNodes.remove(c)

		assert len(set(node.children()).intersection(self.__topLevelNodes)) == 0


	def deregisterNode(self, node):
		'''
		De-registers a Node from the Network.
		This should only be called when deleting a Node.
		This also handles updating the link registration
		in the event that the Node was formed from contracting
		a Link.
		'''

		self.__nodes.remove(node)
		self.__topLevelNodes.remove(node)

		children = node.children()

		self.__topLevelNodes.update(children)

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
		idd = self.__idCounter
		self.__idCounter += 1
		return idd


	def addNodeFromArray(self, arr):
		'''
		Takes as input an array and constructs a Tensor and Node around it,
		then adds the Node to this Network.
		'''
		t = Tensor(arr.shape, arr)
		return Node(t, self, Buckets=[Bucket(self) for _ in range(len(arr.shape))])

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

		link = self.__sortedLinks.pop()

		n1 = link.bucket1().topNode()
		n2 = link.bucket2().topNode()

		n1.merge(n2, mergeL=mergeL, compressL=compressL, eps=eps)


	def linkMerge(self, compressL=False, eps=1e-4):
		'''
		This method checks all Nodes for potential Link mergers and performs any it finds.
		This method takes two keyword arguments:
			compressL	-	Attempts to compress all Links (if any) resulting from a Link merger.
			eps			-	The accuracy of the compression to perform.
		'''
		done = set()
		todo = set(self.__topLevelNodes)

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
		while self.__sortedLinks.length > 0:
			self.merge(mergeL=True, compressL=True)

			if counter%20 == 0:
				t = self.largestTopLevelTensor()
				print(len(self.topLevelNodes()),self.topLevelSize(), t.tensor().shape())
			counter += 1

	def optimize(self, mergeL=True, compressL=True, eps=1e-4):
		todo = set(self.__bottomLevelNodes)
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
				continue
			else:
				optimized = False
				for b in n.buckets():
					if b.linked():
						link = b.link()
						numC = len(link.children())
						if b.numNodes() == 1 and numC > 0:
							# Means that the Node was generated
							# by compressing or merging this Link.
							# Note that there can be at most one such Link
							# for any Node, so we don't mind if the loop
							# continues after we compress.
							if link.compressed() and not link.optimized():
								n1 = n
								n2 = b.otherNodes()[0]

								n11 = n1.children()[0]
								n22 = n2.children()[0]

								done.add(n11)
								done.add(n22)

								doneNodes = []

								_, arr, bs = self.view([n11, n22], mergeL=mergeL, compressL=compressL, eps=eps)

								print(len(n11.linksConnecting(n22)))

								if numC == 1:
									# Means we just compressed a single Link
									prevLink = link.children()[0]

								n1.delete() # We only need to delete one of them, as this deletes the other.

								if numC > 1:
									# Means we've compressed a multiple Links at once
									n1, n2, prevLink = mergeLinks(n11, n22, compressLink=False)
									done.add(n1)
									done.add(n2)


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

	def view(self, nodes, mergeL=True, compressL=True, eps=1e-4):
		'''
		This method calculates the effective Tensor represented by the Network connecting to the given Nodes.

		The way we do this is straightforward: we go through marking Nodes as rejected
		as if we were deleting them. This includes following compression/merger bonds
		where appropriate. We then copy the highest-level Nodes remaining into a new
		Network and contract it. Note that if any specified Nodes are not bottom-level we
		have to pre-reject all of their children.

		This method takes three keywork arguments:
			mergeL 	  - 	If the merger results in a Node which has multiple Links in common with
						another Node, the Links will be merged.
			compressL -	Attempts to compress all Links (if any) resulting from a Link merger.
			eps		  -	The accuracy of the compression to perform.
		'''
		todo = set(nodes)
		rejected = set()
		for n in todo:
			rejected.update(n.allNChildren())

		while len(todo) > 0:
			n = todo.pop()

			if n not in rejected:
				rejected.add(n)
				if n.parent() is not None:
					todo.add(n.parent())

				if len(n.children()) == 1:
					for b in n.buckets():
						link = b.link()
						bo = link.otherBucket(b)
						no = bo.bottomNode()
						if len(no.children()) == 1:
							todo.add(no)

		new1 = set(self.__nodes).difference(rejected)
		new2 = set(self.__nodes).difference(rejected)

		for n in new1:
			if n.parent() in new1:
				new2.remove(n)

		conn = set()
		for n in new1:
			conn.update(n.connected())
		assert len(conn.intersection(nodes)) > 0

		# Build the new Network
		nn = Network()

		# Build the new Nodes
		oldIDs = {}	#	Returns the old Node corresponding to the given new ID.
		newIDs = {}	#	Returns the new Node corresponding to the given old ID.
		newBuckets = {} # Returns the new Bucket corresponding to a given (ID, index) pair in the old Network.
		oldBuckets=  {} # Returns the old Bucket corresponding to a given (ID, index) pair in the new Network.

		newNodes = set()

		for n in new2:
			n1 = nn.addNodeFromArray(n.tensor().array())

			oldIDs[n1.id()] = n
			newIDs[n.id()] = n1

			newNodes.add(n1)

			for i,b in enumerate(n.buckets()):
				newBuckets[(n.id(),i)] = n1.buckets()[i]
				oldBuckets[(n1.id(),i)] = n.buckets()[i]

		# Link up new Nodes
		done = set()

		for n1 in new2:
			n1new = newIDs[n1.id()]
			for ind0, b in enumerate(n1.buckets()):
				if b.linked():
					otherB = b.otherBucket()
					intersection = set(otherB.nodes()).intersection(new2)
					if len(intersection) > 0:
						n2 = intersection.pop()
						ind1 = n2.buckets().index(otherB)
						n2new = newIDs[n2.id()]
						if (n1new.id(), ind0, n2new.id(), ind1) not in done:
							n1new.addLink(n2new, ind0, ind1)
							done.add((n1new.id(), ind0, n2new.id(), ind1))
							done.add((n2new.id(), ind1, n1new.id(), ind0))

		# Check connectivity
		for n in nn.topLevelNodes():
			for b in n.buckets():
				if b.linked():
					assert oldIDs[b.otherTopNode().id()] not in nodes
				else:
					assert len(set(oldIDs[n.id()].connected()).intersection(nodes)) > 0


		# Contract new Network
		nn.contract(mergeL=mergeL, compressL=compressL, eps=eps)

		# Build contracted Tensor and bucketList
		arr = np.array([1.])
		bucketList = []

		originalBuckets = set()
		for n in nodes:
			originalBuckets.update(n.buckets())

		for n in nodes:
			print(n)
		print('---')

		for n in nn.topLevelNodes():
			print(n)

		print('---')
		for n in nn.topLevelNodes(): # Go through all top-level Nodes in the new Network
			arr = np.tensordot(arr, n.tensor().array(), axes=0) # Outer product our Tensor with this one
			for b in n.buckets(): # Go through all Buckets in this Node
				nb = b.bottomNode()	 # Find the bottom Node on each Bucket
				ind = nb.bucketIndex(b) # Find the index corresponding to this Bucket on that Node
				originalExternalBucket = oldBuckets[(nb.id(),ind)]
				for p in originalExternalBucket.topNode().connected():
					print('p',p)
				originalInternalBucket = originalExternalBucket.otherBucket()
				bucketList.append(originalInternalBucket) # Add the corresponding Bucket to our list
				for q in originalExternalBucket.otherNodes():
					print(q)
				print('---')
				assert len(set(originalExternalBucket.otherNodes()).intersection(nodes))>0
				assert originalInternalBucket in originalBuckets

		return nn, arr[0], bucketList





