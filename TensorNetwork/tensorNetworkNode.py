class Link:
	'''
	A Link is a means of indicating an intent to contract two tensors.
	This object has two Buckets, one for each TensorNetworkNode being connected.
	In addition, it has a method for computing the von Neumann entropy of the Link.
	In cases where this computation is intractable due to memory requirements, a heuristic
	is used.

	Links have the following functions:

	bucket1		-	Returns the first Bucket this link connects to.
	bucket2		-	Returns the second Bucket this link connects to.
	entropy		-	Returns the entropy of the Link. Heuristics are used where memory requires them.
	delete		-	Removes this link from both 

	Links are instantiated with the buckets they connect, and are added to the end of the Link
	lists of their buckets. They are also added to the link registry of their TensorNetwork.
	'''

	def __init__(self, b1, b2, network):
		self.__b1 = b1
		self.__b2 = b2
		self.__network = network
		self.__network.registerLink(self)

	def bucket1(self):
		return self.__b1

	def bucket2(self):
		return self.__b2

	def entropy(self):
		raise NotImplementedError

	def delete(self):
		self.__network.deregisterLink(self)
		self.__b1.removeLink(self)
		self.__b2.removeLink(self)

class Bucket:
	'''
	A Bucket is a means of externally referencing an index of a tensor which handles the way in which
	the tensor is linked to other tensors.

	Each Bucket references exactly one index, but may contain multiple Links between that index and others.
	This allows a given tensor to be part of a heirarchical network, wherein nodes may be merged while
	retaining information about the unmerged structure. To accomodate this, each Bucket contains a list
	of Links. When two nodes are merged, the old links remain, while a new Link to the merged object
	is added to the end of the Link list.

	Buckets have the following functions:

	node 		-	Returns the TensorNetworkNode this Bucket belongs to.
	index 		-	Returns the index of the TensorNetworkNode's Tensor this Bucket refers to.
	network 	-	Returns the TensorNetwork this Bucket belongs to.
	numLinks	-	Returns the number of Links this Bucket has.
	link 		-	Takes as input an integer specifying the index of the Link of interest and returns
					that link.
	otherBucket	-	Takes as input an integer specifying the index of the Link of interest and returns
					the Bucket on the other side of that Link.
	otherNode	-	Takes as input an integer specifying the index of the Link of interest and returns
					the TensorNetworkNode on the other side of that Link.
	addLink		-	Takes as input a Link and appends it to the end of the Link list.
	removeLink	-	Removes a Link from the Link list. Raises a ValueError if the Link is not present.
	'''

	def __init__(self, node, index, network):
		self.__node = node
		self.__index = index
		self.__network = network

		self.__links = None

	def node(self):
		return self.node

	def index(self):
		return self.index

	def network(self):
		return self.network

	def numLinks(self):
		return len(self.__links)

	def link(self, index):
		return self.__links[index]

	def otherBucket(self, index):
		b = self.__links[index].bucket1
		if b == self:
			b = self.__links[index].bucket2
		return b

	def otherNode(self, index):
		return self.otherBucket(index).node()

	def addLink(self, link):
		self.__links.append(link)

	def removeLink(self, link):
		if link in self.__links:
			self.__links.remove(link)
		else:
			raise ValueError



class Bucket:
	def __init__(self, tensor, index, all_links):
		self.tensor = tensor
		self.index = index

		self.all_links = all_links

		self.link = None
		self.otherTensor = None
		self.otherBucket = None

	def makeLink(self, other):
		self.otherBucket = other

		self.link = Link(self,other,self.all_links)
		other.link = self.link

		other.otherBucket = self

		self.otherTensor = other.tensor
		other.otherTensor = self.tensor

		self.tensor.connected[self.otherTensor].append(self.link)
		other.tensor.connected[other.otherTensor].append(self.link)

class Tensor:
	def __init__(self, array, all_links, children = None, parent = None, kind = None, idd = None, network = None):
		self.array = np.copy(array)

		self.buckets = []

		for i in range(len(self.array.shape)):
			self.buckets.append(Bucket(self,i,all_links))

		self.connected = defaultdict(list)

		self.all_links = all_links
		self.children = children
		self.parent = parent
		self.kind = kind
		self.id = idd
		self.network = network

	def __str__(self):
		s = 'Tensor: '+str(self.id)
		s = s + '  Shape:'+str(self.array.shape)+'\n'
		s = s + '  Connections:\n'
		for i,b in enumerate(self.buckets):
			if b.link is not None:
				indexSelf = str(i)
				indexOther = str(b.otherTensor.buckets.index(b.otherBucket))
				s = s + 'Index ' + indexSelf + ' to ID ' + str(b.otherTensor.id)
				s = s + ' with index ' + indexOther + '\n'
		return s

	def swapIndices(self, i, j):
		self.buckets[i], self.buckets[j] = self.buckets[j], self.buckets[i]
		self.array = np.swapaxes(self.array,i,j)

	def mergeLinks(self, other):
		links = self.connected[other]
		if len(links) >= 2:
			lenlinks = len(links)
			for i in range(len(links)):
				ind = None
				if links[i].bucket1.tensor == self:
					ind = self.buckets.index(links[i].bucket1)
				else:
					ind = self.buckets.index(links[i].bucket2)
				self.swapIndices(i,ind)

				ind = None
				if links[i].bucket1.tensor == other:
					ind = other.buckets.index(links[i].bucket1)
				else:
					ind = other.buckets.index(links[i].bucket2)
				other.swapIndices(i,ind)

			self.array = np.reshape(self.array,[-1] + list(self.array.shape[len(links):]))
			other.array = np.reshape(other.array,[-1] + list(other.array.shape[len(links):]))

			for i in range(len(links)-1):
				self.buckets[i].link.delete()

			self.buckets = self.buckets[lenlinks-1:]
			other.buckets = other.buckets[lenlinks-1:]

	def mergeAllLinks(self):
		for t in self.connected.keys():
			self.mergeLinks(t)

	def trace(self, ind0, ind1):
		self.array = np.trace(self.array, axis1=ind0, axis2=ind1)

		b0 = self.buckets[ind0]
		b1 = self.buckets[ind1]

		self.buckets.remove(b0)
		self.buckets.remove(b1)

		del b0
		del b1

	def addLink(self, other, indSelf, indOther, kind='outside'):
		# If kind is outside then indSelf and indOther are assumed to refer to outside (original) indices.
		# Otherwise they are inside indices.
		if kind=='outside':
			for q in range(len(self.buckets)):
				if self.buckets[q].index == indSelf:
					i = q
			for q in range(len(other.buckets)):
				if other.buckets[q].index == indOther:
					j = q
		else:
			i = indSelf
			j = indOther

		if self.buckets[i].link is not None:
			raise ValueError('Error: That bucket is already occupied.')
		if other.buckets[j].link is not None:
			raise ValueError('Error: That bucket is already occupied.')

		if self == other:
			self.trace(i,j)
		else:
			b1 = self.buckets[i]
			b2 = other.buckets[j]

			# Build a link
			b1.makeLink(b2)
			l = b1.link

	def contract(self, other, reshape=True): # There should be just one link
		if other not in self.connected:
			raise ValueError('Tensors not connected!')
		else:
			self.mergeLinks(other)
			link = self.connected[other][0]
			bSelf = link.bucket1
			bOther = link.bucket2

			indSelf = self.buckets.index(bSelf)
			indOther = other.buckets.index(bOther)

			t = np.tensordot(self.array, other.array, axes=((indSelf,),(indOther,)))

			if reshape:
				prodSelf = [self.array.shape[j] for j in range(len(self.array.shape)) if j != indSelf]
				prodOther = [other.array.shape[j] for j in range(len(other.array.shape)) if j != indOther]

				prodSelf = np.prod(prodSelf)
				prodOther = np.prod(prodOther)

				t = np.reshape(t,(prodSelf, prodOther))

			return t
