
class Network:
	'''
	A Network is an object storing Nodes as well as providing helper methods
	for manipulating those Nodes.
	'''

	def __init__(self):
		self.nodes = set()

		self.idDict = {}
		self.idCounter = 0

		self.size = 0


	def __str__(self):
		s = 'Network\n'
		for n in self.nodes:
			s = s + str(n) + '\n'
		return s

	def registerNode(self, node):
		'''
		Registers a new Node in the Network.
		This should only be called when registering a new Node.
		'''
		assert node not in self.nodes

		self.nodes.add(node)
		self.idDict[node.id] = node
		self.idCounter += 1

	def deregisterNode(self, node):
		'''
		De-registers a Node from the Network.
		This should only be called when deleting a Node.
		This also handles updating the link registration
		in the event that the Node was formed from contracting
		a Link.
		'''
		assert node in self.nodes

		self.nodes.remove(node)
		self.idDict.pop(node.id)
