





class TensorNetwork:
	def __init__(self):
		self.tensors = set()
		self.all_links = set()
		self.idDict = {}
		self.idCounter = 0
		self.scalar = 1.0
		self.logScalar = 0.0

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

