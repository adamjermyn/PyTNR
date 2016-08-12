from tensor import Tensor
import numpy as np
from scipy.sparse.linalg import svds

def bigSVD(matrix):
	err = 1
	k = 2
	while err > 1e-4:
		u, s, v = svds(matrix, k=k)
		err = np.sum(np.abs(matrix - np.einsum('ij,j,jk->ik',u,s,v)))/np.sum(np.abs(matrix))
		k += 4
	print 'k=',k-4,np.abs(s)/np.sum(s)
	return u, s, v

def compress(link, eps=1e-4):
	n1 = link.bucket1().topNode()
	n2 = link.bucket2().topNode()

	t1 = n1.tensor()
	t2 = n2.tensor()

	arr1 = t1.array()
	arr2 = t2.array()

	ind1 = n1.bucketIndex(link.bucket1())
	ind2 = n2.bucketIndex(link.bucket2())

	cont = t1.contract(ind1,t2,ind2)
	arrN = cont.array()

	sh1 = list(arr1.shape)
	sh2 = list(arr2.shape)

	sh1m = sh1[:ind1] + sh1[ind1+1:]
	sh2m = sh2[:ind2] + sh2[ind2+1:]

	arrN = np.reshape(arrN,(np.product(sh1m),np.product(sh2m)))

#	u, lam, v = bigSVD(arrN)
	u, lam, v = np.linalg.svd(arrN,full_matrices=0)

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	ind = np.searchsorted(cp, eps, side='left')
	ind = len(cp) - ind

	if ind == len(cp): 	# Means we won't actually compress it
		link.setCompressed()
		return link, n1, n2
	else:				# Means we will compress it
		u = np.transpose(u)

		lam = lam[:ind]
		u = u[:ind,:]
		v = v[:ind,:]

		u *= np.sqrt(lam[:,np.newaxis])
		v *= np.sqrt(lam[:,np.newaxis])

		if ind > 1:
			u = np.reshape(u,[ind] + sh1m)
			v = np.reshape(v,[ind] + sh2m)

			perm1 = range(len(arr1.shape))
			perm1 = perm1[1:]
			perm1.insert(ind1,0)
			perm2 = range(len(arr2.shape))
			perm2 = perm2[1:]
			perm2.insert(ind2,0)

			u = np.transpose(u,axes=perm1)
			v = np.transpose(v,axes=perm2)

			t1m = Tensor(u.shape,u)
			t2m = Tensor(v.shape,v)

			n1m = n1.modify(t1m, repBuckets=[ind1])
			n2m = n2.modify(t2m, repBuckets=[ind2])

			newLink = n1m.addLink(n2m, ind1, ind2, compressed=True, children=[link])
		else:	# Means we're just cutting the bond
			u = np.reshape(u,sh1m)
			v = np.reshape(v,sh2m)

			t1m = Tensor(u.shape,u)
			t2m = Tensor(v.shape,v)

			n1m = n1.modify(t1m, delBuckets=[ind1])
			n2m = n2.modify(t2m, delBuckets=[ind2])

			# These four lines make me a bit nervous. In particular,
			# there should be a more elegant way to say that this link is not top-level without
			# incurring infinite recursion in the inheritance. Not that I mind, but doing this
			# could introduce obnoxious edge cases if I'm not careful. It might be better to have
			# an explicit list of no-parent-yet-top cases and just exclude them from top-level
			# things... but that might be slow. Anyway, I also can't quite figure
			# out whether or not the update call is necessary.
			link.setCompressed()
			link.setParent(link) # So it isn't considered top-level
			link.network().deregisterLinkTop(link) # So it's removed from the top-level.
			link.update() # So it's up to date.

			newLink = link

	return newLink, n1m, n2m