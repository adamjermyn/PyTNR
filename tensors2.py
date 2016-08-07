import numpy as np
import networkx
from collections import Counter

class Bucket:
	def __init__(self, tensor, other, link, index):
		self.tensor = tensor
		self.link = link
		self.index = index
		self.other = other # Other Tensor

		self.otherBucket = link.bucket1
		if self.otherBucket == self:
			self.otherBucket = link.bucket2

	def removeIndex(self, i):
		if self.index > i:
			self.index -= 1

	def addIndices(self, num):
		self.index += num

	def moveTensor(self, tensor, index):
		# Deregister other side of link
		self.other.connected.remove(self.tensor)

		# Reregister other side of link
		self.other.connected.add(tensor)

		# Deregister self side of link
		self.tensor.connected.remove(self.other)
		self.tensor.buckets.remove(self)

		# Set new tensor
		self.tensor = tensor

		# Set new index
		self.index = index

		# Register self side of link
		self.tensor.connected.add(self.other)
		self.tensor.buckets.append(self)

		# Register with partner bucket
		self.otherBucket.other = self.tensor

class Link:
	def __init__(self, bucket1, bucket2, all_links):
		self.bucket1 = bucket1
		self.bucket2 = bucket2
		self.all_links = all_links

		self.all_links.add(self)

	def __str__(self):
		s = str(self.bucket1.tensor.id) + ','+str(self.bucket1.index) + ','
		s = s + str(self.bucket2.tensor.id) + ','+str(self.bucket2.index)
		return s

	def delete(self):
		# Disconnect tensors
		self.bucket1.tensor.connected.remove(self.bucket1.other)
		if self.bucket1.tensor != self.bucket2.tensor:
			self.bucket2.tensor.connected.remove(self.bucket2.other)

		# Remove buckets
		self.bucket1.tensor.buckets.remove(self.bucket1)
		self.bucket2.tensor.buckets.remove(self.bucket2)
		del self.bucket1
		del self.bucket2

		# De-register link
		self.all_links.remove(self)

		# Delete
		del self		

class Tensor:
	def __init__(self, array, all_links, children = None, parent = None, kind = None, idd = None, network = None):
		self.buckets = []
		self.connected = set()
		self.indexDict = {}

		# outsideIndices stores which inside index corresponds to a given outside index
		self.outsideIndices = list(range(len(array.shape)))
		self.array = np.copy(array)
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
		if len(self.connected) > 0:
			for b in self.buckets:
				bb = list(filter(lambda x: x.other == self, b.other.buckets))[0]
				s = s + 'Index '+str(b.index)+' to ID '+str(b.other.id)+' with index '+str(bb.index) + '\n'
		return s

	def swapIndices(self, i, j):
		self.array = np.swapaxes(self.array,i,j)
		b0 = self.indexDict.get(i)
		b1 = self.indexDict.get(j)
		if b0 is not None:
			b0.index = j
			self.indexDict.pop(i)
			self.indexDict[j] = b0
		if b1 is not None:
			b1.index = i
			self.indexDict.pop(j)
			self.indexDict[i] = b1

	def mergeIndices(self, i, j):
		# Merges indices but not buckets
		self.swapIndices(0,i)
		self.swapIndices(1,j)
		self.array = np.reshape(self.array,[-1]+ self.array.shape[2:])
		self.indexDict[1].index = 0



	def findBucket(self, other):
		# Find the existing bucket of self (of which only zero or one should exist)
		b = list(filter(lambda x: x.other == other, self.buckets))

		if len(b) == 0:
			return None
		else:
			return b[0]

	def trace(self, ind0, ind1):
		self.array = np.trace(self.array, axis1=ind0, axis2=ind1)
		for b in self.buckets:
			sub = 0
			if b.index > ind0:
				sub += 1
			if b.index > ind1:
				sub += 1
			b.index -= sub
		for q in range(len(self.outsideIndices)):
			sub = 0
			if self.outsideIndices[q] > ind0:
				sub += 1
			if self.outsideIndices[q] > ind1:
				sub += 1
			self.outsideIndices[q] -= sub

	def addLink(self, other, indSelf, indOther, kind='outside'):
		# If kind is outside then indSelf and indOther are assumed to refer to outside (original) indices.
		# Otherwise they are inside indices.
		if kind=='outside':
			i = self.outsideIndices[indSelf]
			j = other.outsideIndices[indOther]
		else:
			i = indSelf
			j = indOther

		if len(list(filter(lambda x: x.index == i,self.buckets))) > 0:
			raise ValueError('Error: That bucket is already occupied.')
		if len(list(filter(lambda x: x.index == j,other.buckets))) > 0:
			raise ValueError('Error: That bucket is already occupied.')

		if self == other:
			self.trace(i,j)
		else:
			# Build a link
			l = Link(None, None, self.all_links)

			# Build buckets
			b1 = Bucket(self, other, l, i)
			b2 = Bucket(other, self, l, j)

			# Add buckets to link
			l.bucket1 = b1
			l.bucket2 = b2

			# Fix bucket references
			l.bucket1.otherBucket = l.bucket2
			l.bucket2.otherBucket = l.bucket1

			# Add buckets to tensors
			self.buckets.append(b1)
			other.buckets.append(b2)

			# Add tensors to tensors
			self.connected.add(other)
			other.connected.add(self)

	def mergeIndices(self, b0, b1):
		# Verify that there is a single tensor on the other side of the link
		if b0.other != b1.other:
			raise ValueError('Error: Cannot merge links with different tensors.')

		# Process first tensor

		# Assuming both indices have associated buckets
		i = b0.index
		j = b1.index

		# Move the indices to the beginning of self.array and merge
		perm = list(range(len(self.array.shape)))
		print perm, i, j
		perm.remove(i)
		perm.remove(j)
		perm.insert(0,i)
		perm.insert(0,j)
		self.array = np.transpose(self.array,axes = perm)
		self.array = np.reshape(self.array, [-1] + list(self.array.shape[2:]))

		# Fix other bucket indices
		for b in self.buckets:
			if b.index < min(i,j):
				b.index += 1
			elif b.index > max(i,j):
				b.index -= 1

		for q in range(len(self.outsideIndices)):
			if self.outsideIndices[q] < min(i,j):
				self.outsideIndices[q] += 1
			elif self.outsideIndices[q] > max(i,j):
				self.outsideIndices[q] -= 1

		# Fix merged index
		b0.index = 0

		# Fix outside indexing for merged indices
		self.outsideIndices[i] = 0
		self.outsideIndices[j] = 0

		# Process second tensor

		# Assuming both indices have associated buckets
		b0 = b0.otherBucket
		b1 = b1.otherBucket

		i = b0.index
		j = b1.index

		# Move the indices to the beginning of self.array and merge
		perm = list(range(len(b0.tensor.array.shape)))
		print perm, i, j
		perm.remove(i)
		perm.remove(j)
		perm.insert(0,i)
		perm.insert(0,j)
		b0.tensor.array = np.transpose(b0.tensor.array,axes = perm)
		b0.tensor.array = np.reshape(b0.tensor.array, [-1] + list(b0.tensor.array.shape[2:]))

		# Fix other bucket indices
		for b in b0.tensor.buckets:
			if b.index < min(i,j):
				b.index += 1
			elif b.index > max(i,j):
				b.index -= 1

		for q in range(len(b0.tensor.outsideIndices)):
			if b0.tensor.outsideIndices[q] < min(i,j):
				b0.tensor.outsideIndices[q] += 1
			elif b0.tensor.outsideIndices[q] > max(i,j):
				b0.tensor.outsideIndices[q] -= 1

		# Fix merged index
		b0.index = 0

		# Fix outside indexing for merged indices
		b0.tensor.outsideIndices[i] = 0
		b0.tensor.outsideIndices[j] = 0

		# Delete link
		b1.link.delete()

		# Ensure that tensors are still connected
		self.connected.add(b0.tensor)
		b0.tensor.connected.add(self)

	def bulkMergeIndices(self):
		counter = Counter([b.other for b in self.buckets])
		for t in self.connected:
			if counter[t] > 1:
				buckets = list(b for b in self.buckets if b.other == t)
				while len(buckets) > 1:
					b0 = buckets[0]
					b1 = buckets[1]
					buckets.remove(b0)
					buckets.remove(b1)
					self.mergeIndices(b0,b1)

	def contract(self, other, reshape=True):
		if other not in self.connected:
			raise ValueError('Tensors not connected!')
		else:
			bSelf = self.findBucket(other)
			bOther = other.findBucket(self)

			t = np.tensordot(self.array, other.array, axes=((bSelf.index,),(bOther.index,)))

			if reshape:
				prodSelf = [self.array.shape[j] for j in range(len(self.array.shape)) if j != bSelf.index]
				prodOther = [other.array.shape[j] for j in range(len(other.array.shape)) if j != bOther.index]

				prodSelf = np.prod(prodSelf)
				prodOther = np.prod(prodOther)

				t = np.reshape(t,(prodSelf, prodOther))

			return t

class TensorNetwork:
	def __init__(self):
		self.tensors = set()
		self.all_links = set()
		self.idDict = {}
		self.idCounter = 0

	def __str__(self):
		s = ''
		for t in self.tensors:
			s = s + str(t)
		return s

	def deepcopy(self):
		network = TensorNetwork()
		tensors = {}

		for t in self.tensors:
			tens = network.addTensor(t.array,kind=t.kind)
			tens.id = t.id
			tens.outsideIndices = list(t.outsideIndices)
			tensors[t.id] = tens

		network.idDict.clear()

		for t in self.tensors:
			network.idDict[t.id] = tensors[t.id]

		for l in self.all_links:
			print l.bucket1.index, l.bucket2.index
			print tensors[l.bucket1.tensor.id]
			print tensors[l.bucket2.tensor.id]
			tensors[l.bucket1.tensor.id].addLink(tensors[l.bucket2.tensor.id],l.bucket1.index,l.bucket2.index,kind='inside')

		return network

	def addTensor(self, array, children=None, parent=None, kind=None):
		t = Tensor(array, self.all_links, children=children,parent=parent,kind=kind,idd=self.idCounter,network=self)
		self.tensors.add(t)
		self.idDict[self.idCounter] = t
		self.idCounter += 1
		return t

	def graph(self):
		G = networkx.Graph()

		for link in self.all_links:
			G.add_edge(link.bucket1.tensor.id,link.bucket2.tensor.id,weight=self.entropy(link=link))

		return G

	def entropy(self, link=None):
		if link is None:
			s = np.zeros(len(self.all_links))
			for i,link in enumerate(self.all_links):
				t = link.bucket1.tensor.contract(link.bucket2.tensor)
				v, lam, u = np.linalg.svd(t)
				lam /= np.sum(lam)
				lam *= lam
				s[i] = -np.sum(lam*np.log(lam))
			return s
		else:
			t = link.bucket1.tensor.contract(link.bucket2.tensor)
			v, lam, u = np.linalg.svd(t)
			lam /= np.sum(lam)
			lam *= lam
			s = -np.sum(lam*np.log(lam))
			return s

	def mergeNodes(self, link, prevNetwork=None):
		t1 = link.bucket1.tensor
		t2 = link.bucket2.tensor

		# Verify that nodes have same kind
		if t1.kind != t2.kind:
			raise ValueError('Error: Cannot merge tensors of different kinds.')
		if t1 == t2:
			raise ValueError('Error: Cannot merge nodes to themselves.')

		# Contract tenors
		t = t1.contract(t2,reshape=False)

		# Create link list
		links = set()
		for b in t1.buckets:
			if b.other is not t2:
				links.add(b.link)
		for b in t2.buckets:
			if b.other is not t1:
				links.add(b.link)

		# Create and register new tensor
		children = None
		if prevNetwork is not None:
			children = [prevNetwork.idDict[t1.id],prevNetwork.idDict[t2.id]]
		tens = self.addTensor(t,children=children,kind=link.bucket1.tensor.kind)
		if prevNetwork is not None:
			prevNetwork.idDict[t1.id].parent = tens
			prevNetwork.idDict[t2.id].parent = tens

		# Move all links to new tensor
		for b in t1.buckets:
			if b.link != link:
				ind = b.index
				if b.index > link.bucket1.index:
					ind -= 1
				b.moveTensor(tens,ind)
		for b in t2.buckets:
			if b.link != link:
				ind = b.index + len(t1.array.shape)
				if ind > link.bucket2.index:
					ind -= 2
				else:
					ind -= 1
				b.moveTensor(tens,ind)

		# Ignore outside indices... irrelevant now that we've merged
		tens.outsideIndices = list(range(len(tens.array.shape)))

		# Merge links
		print tens.network
		tens.bulkMergeIndices()

		# De-register tensors
		self.idDict.pop(t1.id)
		self.idDict.pop(t2.id)

		# Remove link
		link.delete()

		# Remove tensors
		self.tensors.remove(t1)
		self.tensors.remove(t2)
		del t1
		del t2

		return tens

	def planMerger(self):
		graphs = [self.graph()]
		counter = 1
		while counter != 0:
			counter = 0
			graphs2 = []
			for g in graphs:
				if len(g) > 2:
					minCut = networkx.algorithms.connectivity.stoerwagner.stoer_wagner(g,weight='weight')
					graphs2.append(g.subgraph(minCut[1][0]))
					graphs2.append(g.subgraph(minCut[1][1]))
					counter += 1
				else:
					graphs2.append(g)
			if counter > 0:
				graphs = graphs2
		return graphs

	def merge(self):
		copy = self.deepcopy()
		plan = self.planMerger()

		print map(str,self.all_links)
		print map(str,copy.all_links)
		print copy

		print 'abababa'
		for graph in plan:
			if len(graph) > 1:
				t1,t2 = graph.nodes()
				t1 = copy.idDict[t1]
				t2 = copy.idDict[t2]
				b = t1.findBucket(t2)
				link = b.link
				copy.mergeNodes(link,prevNetwork=self)
				print t1.id, t2.id
				print map(str,self.all_links)
				print map(str,copy.all_links)
				print copy
				print '-------------'
		return copy