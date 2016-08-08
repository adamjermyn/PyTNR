





class TensorNetwork:
	def __init__(self):
		self.tensors = set()
		self.all_links = set()
		self.idDict = {}
		self.idCounter = 0
		self.scalar = 1.0
		self.logScalar = 0.0

	def addTensor(self, array, children=None, parent=None, kind=None):
		t = Tensor(array, self.all_links, children=children,parent=parent,kind=kind,idd=self.idCounter,network=self)
		self.tensors.add(t)
		self.idDict[self.idCounter] = t
		self.idCounter += 1
		return t

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

