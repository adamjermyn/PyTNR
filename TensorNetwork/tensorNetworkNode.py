class Bucket:
	'''
	A bucket is a means of externally referencing an index of a tensor which handles the way in which
	the tensor is linked to other tensors.

	Each Bucket references exactly one index, but may contain multiple Links between that index and others.
	This allows a given tensor to be part of a heirarchical network, wherein nodes may be merged while
	retaining information about the unmerged structure. To accomodate this, each TensorNetworkNode is
	associated with an integer known as a level, specifying where it sits in the structure. Each Link also
	has such a number, and the level corresponding to that Link is the greater of the two node levels on
	either end of the link.

	Link is associated with
	an integer known as the link level. The link level must be the greater of the 



	two integers, corresponding to a link level on either side. For instance, suppose that three nodes A, B,
	and C are all connected pairwise. Each of these links will have levels (0,0). If we merge A and B, a new
	link will exist between this merged object and C, with level 1 for the merged object and level 0 for the 
	are linked 

	Buckets have the following functions:

	node 		-	Returns the TensorNetworkNode this bucket belongs to.
	index 		-	Returns the index of the TensorNetworkNode's Tensor this bucket refers to.
	network 	-	Returns the TensorNetwork this bucket belongs to.
	links 		-	Takes as input an integer specifying a link level and returns all links of that level.
					There may be at most a single link of each level. If a link exists at level n then one
					necessarily exists at level n-1 for all n > 1.
	otherBucket	-	Takes as input an integer specifying a link level and returns the bucket on the other
					side of the link of that level, if there is one. If there isn't one this raises an
					AttributeError.
	otherNode	-	Takes as input an integer specifying a link level and returns the TensorNetworkNode
					on the other side of the link of that level, if there is one. If there isn't one
					this raises an AttributeError.
	makeLink	-	Takes as input another Bucket and creates a Link between this Bucket and that one.
					The link level is automatically assigned to be 
	'''

	def __init__(self, node, index, network):
		self.__node = node
		self.__index = index
		self.__network = network

		self.links = None

	def node(self):
		return self.node

	def index(self):
		return self.index

	def network(self):
		return self.network



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


class Link:
	def __init__(self, bucket1, bucket2, all_links):
		self.bucket1 = bucket1
		self.bucket2 = bucket2
		self.all_links = all_links

		self.entropy = None

		self.all_links.add(self)

	def __str__(self):
		s = '('+str(self.bucket1.tensor.id) + ','+str(self.bucket1.index) + '),('
		s = s + str(self.bucket2.tensor.id) + ','+str(self.bucket2.index)+')'
		return s

	def delete(self):
		# De-register link
		self.all_links.remove(self)

		# Deregister from tensors
		self.bucket1.tensor.connected[self.bucket2.tensor].remove(self)
		self.bucket2.tensor.connected[self.bucket1.tensor].remove(self)

		# Disconnect buckets
		self.bucket1.otherTensor = None
		self.bucket2.otherTensor = None
		self.bucket1.otherBucket = None
		self.bucket2.otherBucket = None
		self.bucket1.link = None
		self.bucket2.link = None

		# Delete
		del self		

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
