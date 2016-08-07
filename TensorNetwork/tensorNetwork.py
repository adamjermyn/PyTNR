import numpy as np
import networkx
from collections import Counter
from collections import defaultdict
#import pymetis
import itertools as it

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))






class TensorNetwork:
	def __init__(self):
		self.tensors = set()
		self.all_links = set()
		self.idDict = {}
		self.idCounter = 0
		self.scalar = 1.0
		self.logScalar = 0.0

	def __str__(self):
		s = ''
		for t in self.tensors:
			s = s + str(t) + '\n'
		return s

	def addTensor(self, array, children=None, parent=None, kind=None):
		t = Tensor(array, self.all_links, children=children,parent=parent,kind=kind,idd=self.idCounter,network=self)
		self.tensors.add(t)
		self.idDict[self.idCounter] = t
		self.idCounter += 1
		return t

	def deepcopy(self):
		network = TensorNetwork()
		tensors = {}


		for t in self.tensors:
			tens = network.addTensor(t.array,kind=t.kind)
			tens.id = t.id
			tensors[t.id] = tens

		network.idDict.clear()

		for t in self.tensors:
			network.idDict[t.id] = tensors[t.id]

		for l in self.all_links:
			t1 = tensors[l.bucket1.tensor.id]
			t2 = tensors[l.bucket2.tensor.id]
			ind1 = l.bucket1.tensor.buckets.index(l.bucket1)
			ind2 = l.bucket2.tensor.buckets.index(l.bucket2)
			t1.addLink(t2,ind1,ind2,kind='inside')

		network.idCounter = self.idCounter
		network.scalar = self.scalar
		network.logScalar = self.logScalar

		return network

	def graph(self):
		G = networkx.Graph()

		for link in self.all_links:
			s = self.entropy(link=link)
			G.add_edge(link.bucket1.tensor.id,link.bucket2.tensor.id,weight=s)
		return G

	# This should really be a thing that gets stored by the link objects
	# and then updated as needed. There is no good reason to recompute it
	# in full each time. Also worth noting is that we should probably not
	# bother computing the entropy for links between huge tensors, as they
	# will usually not be the oens merged (we can use a reasonable prescription
	# like some fraction of the bond width for that).
	def entropy(self, link=None):
		if link is None:
			s = np.zeros(len(self.all_links))
			links = []
			for i,link in enumerate(self.all_links):
				t = link.bucket1.tensor.contract(link.bucket2.tensor)
				v, lam, u = np.linalg.svd(t)
				lam = np.abs(lam)**2
				lam /= np.sum(lam)
				sBond = -lam*np.log(lam)
				sBond[lam==0] = 0
				s[i] = np.sum(sBond)
				links.append(link)
			return s,links
		else:
			t = link.bucket1.tensor.contract(link.bucket2.tensor)
			v, lam, u = np.linalg.svd(t)
			lam = np.abs(lam)**2
			lam /= np.sum(lam)
			sBond = -lam*np.log(lam)
			sBond[lam==0] = 0
			s = np.sum(sBond)
			return s

	def compress(self, eps=1e-4):
		netRed = 0

		s = np.zeros(len(self.all_links))
		for i,link in enumerate(self.all_links):
			t1 = link.bucket1.tensor
			t2 = link.bucket2.tensor

			ind1 = t1.buckets.index(link.bucket1)
			ind2 = t2.buckets.index(link.bucket2)

			if t1.array.shape[ind1] > 1:
				t1.swapIndices(0,ind1)
				t2.swapIndices(0,ind2)

				t = t1.contract(t2,reshape=True)

				v, lam, u = np.linalg.svd(t,full_matrices=0)

				p = lam**2
				p /= np.sum(p)
				cp = np.cumsum(p[::-1])

				ind = np.searchsorted(cp, eps, side='left')
				ind = len(cp) - ind

				netRed += len(lam) - ind

				v = v[:,:ind]
				lam = lam[:ind]
				u = u[:ind,:]

				v *= np.sqrt(lam[np.newaxis,:])
				u *= np.sqrt(lam[:,np.newaxis])

				v = np.transpose(v)
				v = np.reshape(v,[ind]+list(t1.array.shape[1:]))

				u = np.reshape(u,[ind]+list(t2.array.shape[1:]))

				t1.array = v
				t2.array = u

		# Remove indices which have just a single-width bond
		for t in self.tensors:
			i = 0
			while i < len(t.array.shape):
				if t.array.shape[i] == 1:
					b = t.buckets[i]
					if b.link is not None:
						to = b.otherTensor
						indo = to.buckets.index(b.otherBucket)
						to.buckets.remove(b.otherBucket)
						t.buckets.remove(b)
						b.link.delete()
					t.array = np.reshape(t.array,list(t.array.shape[:i])+list(t.array.shape[i+1:]))
					to.array = np.reshape(to.array,list(to.array.shape[:indo])+list(to.array.shape[indo+1:]))
				else:
					i += 1

		# Accumulate disconnected tensors in scalar
		rem = set()
		for t in self.tensors:
			if len(t.buckets) == 0:
				self.scalar *= t.array
				self.logScalar += np.log(np.abs(t.array))
				rem.add(t)

		for t in rem:
			self.tensors.remove(t)

		return netRed

	def splitNode(self, tSplit, prevNetwork=None, eps=1e-4):
		neighbors = tSplit.connected.keys()

		# Identify split most likely to be helpful

		pwr = powerset(neighbors)

		best = []
		bestS = 1e10

		if len(neighbors) <= 1:
			return None

		for p in pwr:
			if len(p) > 0 and len(p) < len(neighbors):
				ts1 = []
				ts2 = []
				for t in p:
					ts1.append(t)
				for t in neighbors:
					if t not in p:
						ts2.append(t)

				s = 0


#				for t in ts1:
#					for tt in t.connected.keys():
#						if tt in ts2:
#							s += self.entropy(link=t.connected[tt][0])
#				s -= np.log(tSplit.array.size)
#
				prod1 = 1
				prod2 = 1
				mid = 1

				sArray1 = []
				sArray2 = []

				bArray1 = []
				bArray2 = []

				for b in tSplit.buckets:
					if b.link is not None:
						if b.otherTensor in ts1:
							bArray1.append(b)
							sArray1.append(tSplit.array.shape[tSplit.buckets.index(b)])
							prod1 *= sArray1[-1]
						else:
							bArray2.append(b)
							sArray2.append(tSplit.array.shape[tSplit.buckets.index(b)])
							prod2 *= sArray2[-1]
				mid = (prod1*prod2)**(1./len(neighbors))

				s += np.log(prod1*mid) + np.log(prod2*mid)
				arr = np.copy(tSplit.array)

				arr = np.reshape(arr, (prod1,prod2))

				v, lam, u = np.linalg.svd(arr,full_matrices=0)

				p = lam**2
				p /= np.sum(p)
				cp = np.cumsum(p[::-1])

				ind = np.searchsorted(cp, eps, side='left')
				ind = len(cp) - ind

				v = v[:,:ind]
				lam = lam[:ind]
				u = u[:ind,:]

				v *= np.sqrt(lam[np.newaxis,:])
				u *= np.sqrt(lam[:,np.newaxis])

				v = np.transpose(v)
				v = np.reshape(v,[ind]+sArray1)

				u = np.reshape(u,[ind]+sArray2)

				s = v.size + u.size
				if s < bestS:
					best = [p,sArray1,sArray2, prod1, prod2, bArray1, bArray2]
					bestS = s
		# Perform split
		p, sArray1, sArray2, prod1, prod2, bArray1, bArray2 = best

		arr = np.copy(tSplit.array)

		arr = np.reshape(arr, (prod1,prod2))

		v, lam, u = np.linalg.svd(arr,full_matrices=0)

		p = lam**2
		p /= np.sum(p)
		cp = np.cumsum(p[::-1])

		ind = np.searchsorted(cp, eps, side='left')
		ind = len(cp) - ind

		v = v[:,:ind]
		lam = lam[:ind]
		u = u[:ind,:]

		v *= np.sqrt(lam[np.newaxis,:])
		u *= np.sqrt(lam[:,np.newaxis])

		v = np.transpose(v)
		v = np.reshape(v,[ind]+sArray1)

		u = np.reshape(u,[ind]+sArray2)

		if v.size + u.size > tSplit.array.size:
			return None

		# Need to fix children
		tNew1 = self.addTensor(v,children=None,parent=None,kind=tSplit.kind)
		tNew2 = self.addTensor(u,children=None,parent=None,kind=tSplit.kind)
#		prevNetwork.idDict[tSplit.id].parent=[tNew1,tNew2]

		tNew1.buckets = bArray1
		tNew2.buckets = bArray2
		# Need to also allocate buckets which are not linked, not yet implemented though...

		for b in bArray1:
			b.tensor = tNew1
			tNew1.connected[b.otherTensor].append(b.link)
			b.otherTensor.connected[tNew1].append(b.link)
			del b.otherTensor.connected[tSplit]
			b.otherBucket.otherTensor = tNew1
		for b in bArray2:
			tNew2.connected[b.otherTensor].append(b.link)
			b.otherTensor.connected[tNew2].append(b.link)
			b.tensor = tNew2
			del b.otherTensor.connected[tSplit]
			b.otherBucket.otherTensor = tNew2

		b1 = Bucket(tNew1,0,self.all_links)
		b2 = Bucket(tNew2,0,self.all_links)
		tNew1.buckets.insert(0,b1)
		tNew2.buckets.insert(0,b2)
		b1.makeLink(b2)



		self.tensors.remove(tSplit)
		del self.idDict[tSplit.id]
		del tSplit


		return tNew1, tNew2

	def mergeNodes(self, link, prevNetwork=None):
		t1 = link.bucket1.tensor
		t2 = link.bucket2.tensor

		# Verify that nodes have same kind
		if t1.kind != t2.kind:
			raise ValueError('Error: Cannot merge tensors of different kinds.')
		if t1 == t2:
			raise ValueError('Error: Cannot merge nodes to themselves.')

		# Merge links
		t1.mergeAllLinks()
		t2.mergeAllLinks()

		# Contract tenors
		t = t1.contract(t2,reshape=False)

		# Create and register new tensor
		children = None
		if prevNetwork is not None:
			children = [prevNetwork.idDict[t1.id],prevNetwork.idDict[t2.id]]
		tens = self.addTensor(t,children=children,kind=link.bucket1.tensor.kind)
		if prevNetwork is not None:
			prevNetwork.idDict[t1.id].parent = tens
			prevNetwork.idDict[t2.id].parent = tens

		# Move all links to new tensor
		tens.buckets = t1.buckets + t2.buckets

		for key, val in t1.connected.items():
			tens.connected[key] = tens.connected[key] + val
		for key, val in t2.connected.items():
			tens.connected[key] = tens.connected[key] + val
		for key, val in tens.connected.items():
			tens.connected[key] = list(set(val))

		delList = []
		moveList = []

		for b in tens.buckets:
			if (b.tensor == t1 and b.otherTensor == t2)  or (b.tensor == t2 and b.otherTensor == t1):
				delList.append(b)
			elif b.link is not None:
				moveList.append(b)

		while len(delList) > 0:
			b = delList[0]
			tens.buckets.remove(b)
			tens.buckets.remove(b.otherBucket)
			tens.connected.pop(t1)
			tens.connected.pop(t2)
			delList.remove(b)
			delList.remove(b.otherBucket)
			b.link.delete()

		for b in moveList:
			if b.tensor == t1:
				b.otherTensor.connected.pop(t1)
			else:
				b.otherTensor.connected.pop(t2)
			b.tensor = tens
			b.otherBucket.otherTensor = tens
			b.otherTensor.connected[tens].append(b.link)


		tens.mergeAllLinks()

		# De-register tensors
		self.idDict.pop(t1.id)
		self.idDict.pop(t2.id)

		# Remove tensors
		self.tensors.remove(t1)
		self.tensors.remove(t2)
		del t1
		del t2

		return tens



	def split(self):
		copy = self.deepcopy()

		todo = list(copy.tensors)

		while len(todo) > 0:
			t = todo[0]
			if t.array.size > 200:
				sp = copy.splitNode(t,prevNetwork=self)
				if sp is not None:
					t1, t2 = sp
					if t1.array.size > 200:
						todo.append(t1)
					if t2.array.size > 200:
						todo.append(t2)
			todo = todo[1:]
		return copy



	def merge(self):
		copy = self.deepcopy()
		plan = self.planMerger()

		for graph in plan:
			if len(graph) > 1:
				t1,t2 = graph
				t1 = copy.idDict[t1]
				t2 = copy.idDict[t2]
				tens = copy.mergeNodes(t1.connected[t2][0],prevNetwork=self)
				for g in plan:
					for i in range(len(g)):
						if g[i] == t1.id or g[i] == t2.id:
							g[i] = tens.id
		return copy

	def planMerger(self):
		s, toConsider = self.entropy()

		for i in range(len(s)):
			l = toConsider[i]
			s[i] *= -1
			s[i] /= np.log(2)
			s[i] += l.bucket1.tensor.array.size*l.bucket2.tensor.array.size
			s[i] -= l.bucket1.tensor.array.size
			s[i] -= l.bucket2.tensor.array.size

			t1 = l.bucket1.tensor
			t2 = l.bucket2.tensor

			c1 = set(t1.connected.keys())
			c2 = set(t2.connected.keys())
			c3 = c1.intersection(c2)

			if len(c3) > 0:
				for t in c3:
					# Slightly overestimates corrections (ignores compounding on the merged tensor)
					# This means it overvalues merging links.

					# Also worth seeing if there's a good way to copy a subset of the network
					# so you can estimate what happens directly
					for l in t.connected[t1]:
						bs = l.bucket1.tensor.array.shape[l.bucket1.tensor.buckets.index(l.bucket1)]
						ts = l.bucket1.tensor.array.size
						s[i] -= ts
						s[i] += ts/np.sqrt(bs)
						s[i] -= l.bucket1.tensor.array.size*l.bucket2.tensor.array.size/np.sqrt(bs)
					for l in t.connected[t2]:
						bs = l.bucket1.tensor.array.shape[l.bucket1.tensor.buckets.index(l.bucket1)]
						ts = l.bucket1.tensor.array.size
						s[i] -= ts
						s[i] += ts/np.sqrt(bs)
						s[i] -= l.bucket1.tensor.array.size*l.bucket2.tensor.array.size/np.sqrt(bs)


		toConsider = [y for x,y in sorted(zip(s,toConsider))]
		s = sorted(s)

		print s[:20]

		ind = 0
		toRemove = []
		while ind < len(toConsider):
			link = toConsider[ind]
			if link not in toRemove:
				for i in range(ind+1,len(toConsider)):
					if toConsider[i].bucket1.tensor is link.bucket1.tensor:
						toRemove.append(toConsider[i])
					elif toConsider[i].bucket1.tensor is link.bucket2.tensor:
						toRemove.append(toConsider[i])
					elif toConsider[i].bucket2.tensor is link.bucket1.tensor:
						toRemove.append(toConsider[i])
					elif toConsider[i].bucket2.tensor is link.bucket2.tensor:
						toRemove.append(toConsider[i])
			ind += 1

		for l in toRemove:
			if l in toConsider:
				toConsider.remove(l)

		toMerge = []
		for l in toConsider:
			toMerge.append([l.bucket1.tensor.id,l.bucket2.tensor.id])

		toMerge = toMerge[:1+int(len(toMerge)/20)]

		return toMerge
'''
		### Try switching back to local merging, now that splitting is allowed

	def planMerger(self):
		todo = [self.graph()]
		done = []

		while len(todo) > 0:
			G = todo[0]

			todo = todo[1:]

			if networkx.is_connected(G):				

				# Pre-filter so that nodes are numbered 0...N-1
				nodeInd = {}
				indNode = {}
				for i,n in enumerate(G.nodes()):
					nodeInd[i] = n
					indNode[n] = i
				adj = []
				for i in range(len(G)):
					adj.append([])
					for j in G[nodeInd[i]]:
						adj[-1].append(indNode[j])

				part = pymetis.part_graph(2,adjacency=adj)

				sub1 = []
				sub2 = []

				for i in range(len(G)):
					if part[1][i] == 1:
						sub1.append(nodeInd[i])
					else:
						sub2.append(nodeInd[i])

				g1 = G.subgraph(sub1)
				g2 = G.subgraph(sub2)

				if len(g1) > 2:
					todo.append(g1)
				else:
					done.append(g1)
				if len(g2) > 2:
					todo.append(g2)
				else:
					done.append(g2)
			else:
				graphs = networkx.connected_component_subgraphs(G)
				graphs = list(graphs)
				for g in graphs:
					if len(g) > 2:
						todo.append(g)
					else:
						done.append(g)

		moreDone = []
		for i,g in enumerate(done):
			if len(g) > 1:
				if not networkx.is_connected(g):
					done[i] = g.subgraph([g.nodes()[0]])
					moreDone.append(g.subgraph([g.nodes()[1]]))

		done = done + moreDone

		for i,g in enumerate(done):
			done[i] = g.nodes()

		return done

	def merge(self):
		copy = self.deepcopy()

		counter = 0
		while counter < 10 and len(copy.tensors) > 1:

			s,links = copy.entropy()

			for i in range(len(s)):
				s[i] = -(links[i].bucket1.tensor.array.size*links[i].bucket2.tensor.array.size)

			i = np.argmax(s)

			t1 = links[i].bucket1.tensor
			t2 = links[i].bucket2.tensor

			copy.mergeNodes(t1.connected[t2][0],prevNetwork=None)

			counter += 1

		return copy
'''


'''
	def planMerger(self):
		used = set()
		unused = set()
		mergers = []

		for t in self.tensors:
			unused.add(t)

		s,links = self.entropy()

		sNew = np.copy(s)

		for i in range(len(s)):
			t1 = links[i].bucket1.tensor
			t2 = links[i].bucket2.tensor

			c1 = set(t1.connected.keys())
			c2 = set(t2.connected.keys())
			c3 = c1.intersection(c2)

			if len(c3) > 0:
				for t in c3:
					for l in t.connected[t1]:
						sNew[i] += l.bucket1.tensor.array.shape[l.bucket1.tensor.buckets.index(l.bucket1)]
					for l in t.connected[t2]:
						sNew[i] += l.bucket1.tensor.array.shape[l.bucket1.tensor.buckets.index(l.bucket1)]


#			s[i] /= np.log(links[i].bucket1.tensor.array.size*links[i].bucket2.tensor.array.size)
#			s[i] /= np.log(links[i].bucket1.tensor.array.shape[links[i].bucket1.tensor.buckets.index(links[i].bucket1)])
#			s[i] /= links[i].bucket1.tensor.array.shape[links[i].bucket1.tensor.buckets.index(links[i].bucket1)]

		while len(unused) > 0:
			i = np.argmax(s)
			link = links[i]

			t1 = link.bucket1.tensor
			t2 = link.bucket2.tensor

			if t1 in unused and t2 in unused:
				used.add(t1)
				used.add(t2)
				unused.remove(t1)
				unused.remove(t2)
				mergers.append([t1.id,t2.id])

				for key, val in t1.connected.items():
					for l in val:
						if l in links:
							ind = links.index(l)
							s = np.delete(s,ind)
							links.remove(l)

				for key, val in t2.connected.items():
					for l in val:
						if l in links:
							ind = links.index(l)
							s = np.delete(s,ind)
							links.remove(l)

			if len(links) == 0:
				for t in unused:
					used.add(t)
					mergers.append([t.id])
				unused = set()

		return mergers

	def merge(self):
		copy = self.deepcopy()
		plan = self.planMerger()

		for graph in plan:
			if len(graph) > 1:
				t1,t2 = graph
				t1 = copy.idDict[t1]
				t2 = copy.idDict[t2]
				copy.mergeNodes(t1.connected[t2][0],prevNetwork=self)
		return copy

'''
