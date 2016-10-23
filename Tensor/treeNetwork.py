from network import Network
from node import Node
from bucket import Bucket
from link import Link
from arrayTensor import ArrayTensor
from utils import entropy, split_chunks, splitArray

maxSize = 100000

class TreeNetwork(Network):
	'''
	A treeNetwork is a special case of a Network in which the Network being represented
	contains no cycles. This allows matrix elements of a treeNetwork to be efficiently
	evaluated.

	As the only quantities which matter are the matrix elements, the treeNetwork may
	refactor itself through singular value decomposition (SVD) to minimize memory use, and
	so no assumptions should be made about the Nodes in this object, just the external
	Buckets.

	Internally all Nodes of a treeNetwork have Tensors of rank at most 3.
	SVD factoring is used to enforce this.
	'''

	def __init__(self, accuracy=1e-4):
		'''
		treeNetworks require an accuracy argument which determines how accurately (in terms of relative error)
		they promise to represent their matrix elements.
		'''
		super().__init__()

		self.accuracy = accuracy

	def pathBetween(self, node1, node2, calledFrom=None):
		'''
		Returns the unique path between node1 and node2.
		This is done by treating node1 as the root of the binary tree and performing a depth-first search.
		Note that this search only iterates through the internal buckets in the network: it will not consider
		nodes in another network.
		'''
		if node1 == node2: # Found it!
			print('found!')
			return [node1]

		if len(self.internalConnected(node1)) == 1 and calledFrom is not None:	# Nothing left to search
			return []

		for c in self.internalConnected(node1): # Search children
			assert c in node1.connectedNodes
			l = node1.findLink(c)
			assert l.bucket1 in node1.buckets or l.bucket2 in node1.buckets
			assert l.bucket1 in c.buckets or l.bucket2 in c.buckets
			assert l.bucket1.link is l
			assert l.bucket2.link is l
			assert node1 in c.connectedNodes
			if c is not calledFrom:
				path = self.pathBetween(c, node2, calledFrom=node1)
				if len(path) > 0: # Means the recursive call found it
					assert c in path
					assert c == path[0]
					assert node1 in c.connectedNodes
					assert node1 in path[0].connectedNodes
					path2 = [node1] + path
					for i in range(len(path2)-1):
						assert path2[i] in path2[i+1].connectedNodes
					return path2

		return []

	def contractNode(self, n):
		'''
		This method adds the node n to this network.
		This node must be at most of rank 3.
		The node may be linked already to members of this network.
		This method handles the logic of removing any loops which arise in the process.

		This method is distinct from addNode in that it does not simply append the node
		to the network. In that sense this method is more specialised and obeys more
		stringent conditions.
		'''
		assert n.tensor.rank <= 3

		connected = []
		for c in n.connectedNodes:
			if c in self.nodes:
				connected.append(c)

		# Because of the assertion, len(connected) <= 3
		# If len(connected) <= 1 there's nothing tricky for us
		# to do, but if len(connected) > 1 we have to
		# eliminate loops and such.
		if len(connected) == 1:
			self.addNode(n)
		elif len(connected) == 2:
			n1 = connected[0]
			n2 = connected[1]
			if n1 == n2:
				self.addNode(n)
				n = self.mergeNodes(n, n1)
				print('Just merging.')
			else:
				# Means there's a loop
				print(n1, n2)
				loop = self.pathBetween(n1, n2)
				print('Checking for loop...')
				if len(loop) > 0:
					print('Loop found. Eliminating...')
					self.addNode(n)
					self.eliminateLoop(loop + [n])
					print('Loop eliminated.')
				else:
					self.addNode(n)
					print('No loop found.')
		elif len(connected) == 3:
			raise NotImplementedError('len(connected) == 3')



	def splitNode(self, node, prevIndex=None):
		'''
		Takes as input a Node and ensures that it is at most rank 3 by splitting it recursively.
		Any indices listed in prevIndex will be ignored in balancing the split, so this may be
		used to enforce an ordering of indices.
		'''
		if node.tensor.rank > 3:
			self.removeNode(node)

			array = node.tensor.array

			if prevIndex is None:
				prevIndex = []

			s = []
			indices = list(range(len(array.shape)))

			for i in prevIndex:
				indices.remove(i)

			for i in indices:
				s.append((i, entropy(array, i)))

			chunks = split_chunks(s, 2)

			chunkIndices = [[i[0] for i in chunks[0]],[i[0] for i in chunks[1]]]

			u, v, prevIndices, indices1, indices2 = splitArray(array, [chunkIndices[0], chunkIndices[1]], ignoreIndex=prevIndex, accuracy=self.accuracy)

			b1 = Bucket()
			b2 = Bucket()
			n1 = Node(ArrayTensor(u), Buckets=[node.buckets[i] for i in indices1] + [b1])
			n2 = Node(ArrayTensor(v), Buckets=[b2] + [node.buckets[i] for i in indices2])
			self.addNode(n1)
			self.addNode(n2)
			l = Link(b1,b2)
			assert b1.size == b2.size

			for b in node.buckets:
				assert (b in n1.buckets) or (b in n2.buckets)

			chunkIndices[1] = []
			for i in range(len(v.shape)):
				if i != 0 and i not in prevIndices:
					chunkIndices[1].append(i)

			if len(v.shape) <= 3:
				return [n2, self.splitNode(n1, prevIndex=[len(u.shape)-1])]

			self.removeNode(n2)
			q, v, prevIndices, indices1, indices2 = splitArray(v, [chunkIndices[1], [0]], ignoreIndex=[0] + prevIndices, accuracy=self.accuracy)
			b1 = Bucket()
			b2 = Bucket()
			n3 = Node(ArrayTensor(q), Buckets=[n2.buckets[i] for i in indices1] + [b1])
			n4 = Node(ArrayTensor(v), Buckets=[b2] + [n2.buckets[i] for i in indices2])
			self.addNode(n3)
			self.addNode(n4)
			l = Link(b1,b2)
			assert b1.size == b2.size

			for b in node.buckets:
				assert (b in n1.buckets) or (b in n2.buckets) or (b in n3.buckets) or (b in n4.buckets)


			return [n4, self.splitNode(n1, prevIndex=[len(u.shape)-1]), self.splitNode(n3, prevIndex=[len(q.shape)-1])]
		else:
			return node

	def eliminateLoop(self, loop):
		'''
		Takes as input a list of Nodes which have been linked in a loop.
		The nodes are assumed to be in linkage order (i.e. loop[i] and loop[i+1] are linked),
		and the list is assumed to wrap-around (so loop[0] and loop[-1] are linked).

		The loop is assumed to be the only loop in the Network.

		The loop is eliminated by iteratively contracting along the loop and factoring out
		extra indices as memory requires. This proceeds until the loop has length 3, and then
		one of the three links is cut via SVD (putting all of that link's entropy in the remaining
		two links).
		'''
		print('Eliminating loop...')
		for n in loop:
			print(n.id,' ',)
		for i in range(len(loop)):
			assert loop[i-1] in loop[i].connectedNodes

		assert len(loop) >= 3

		while len(loop) > 3:
			n1 = loop[1]
			n2 = loop[2]
			ind1 = n1.indexConnecting(loop[0])
			ind2 = n2.indexConnecting(loop[3])
			b1 = n1.buckets[ind1]
			b2 = n2.buckets[ind2]

			n = self.mergeNodes(n1, n2)

			loop.pop(1)

			if n.tensor.size > maxSize and n.tensor.rank > 3:
				nodes = self.splitNode(n, prevIndex=[n.bucketIndex(b1),n.bucketIndex(b2)])
				n = nodes[0] # The ignored indices always end up in the first node

			loop[1] = n

		if loop[1].tensor.rank > 3:
			ind1 = n1.indexConnecting(loop[0])
			ind2 = n2.indexConnecting(loop[2])
			b1 = n1.buckets[ind1]
			b2 = n2.buckets[ind2]
			nodes = self.splitNode(n, prevIndex=[n.bucketIndex(b1),n.bucketIndex(b2)])
			n = nodes[0]
			loop[1] = n

		assert len(loop) == 3

		n1 = loop[0]
		n2 = loop[1]
		n3 = loop[2]
		n = self.mergeNodes(n1, n2)
		n = self.mergeNodes(n, n3)

		if n.tensor.size > maxSize and n.tensor.rank > 3:
			self.splitNode(n)

		for n in self.nodes:
			assert n.tensor.rank <= 3
