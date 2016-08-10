from tensor import Tensor
from node import Node
import numpy as np

def compress(link, tolerance, eps=1e-4):

	### TODO: Just flatten all the non-contraction axes to avoid reordering by mistake.

	'''
	So there's a problem with how this has been implemented.
	The problem is that we generate a new level (which is fine) but then preserve
	links between the previous level and this level *including the link between the two
	nodes that we merge*. That is, if the nodes are A and B and after compression become
	A' and B' then A' still links to B and B' still links to A. This is a problem because
	they've undergone basis changes. If we just delete these connections (which seems reasonable)
	we have a few other issues. The main of these is that it breaks the paradigm of having
	every node see a reduced representation of the rest of the network. One option is to
	simply perform replacement compression, in which case we have to remember to compress
	each link as soon as something changes. That basically just makes it more complicated
	to make changes (the main change of interest is merging, followed by optimizing for linear
	optimization problems), as we have to build the re-compression into the changes... Doing
	this also breaks the notion that Nodes are immutable, though that might not be too big a deal
	because we can just build the compression into any change we make...


	'''

	n1 = link.bucket1().node()
	n2 = link.bucket2().node()

	print n1.id(), n2.id()

	t1 = n1.tensor()
	t2 = n2.tensor()

	arr1 = t1.array()
	arr2 = t2.array()

	print arr1.shape
	print arr2.shape

	ind1 = n1.bucketIndex(link.bucket1())
	ind2 = n2.bucketIndex(link.bucket2())

	cont = t1.contract(ind1,t2,ind2)
	arrN = cont.array()

	sh1 = list(arr1.shape)
	sh2 = list(arr2.shape)

	sh1m = sh1[:ind1] + sh1[ind1+1:]
	sh2m = sh2[:ind2] + sh2[ind2+1:]

	arrNp = np.copy(arrN)

	arrN = np.reshape(arrN,(np.product(sh1m),np.product(sh2m)))

	arrNc = np.copy(arrN)

	u, lam, v = np.linalg.svd(arrN,full_matrices=0)

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	ind = np.searchsorted(cp, eps, side='left')
	ind = len(cp) - ind

	u = u[:,:ind]
	lam = lam[:ind]
	v = v[:ind,:]

	u = np.transpose(u)

	u *= np.sqrt(lam[:,np.newaxis])
	v *= np.sqrt(lam[:,np.newaxis])

	u = np.reshape(u,[ind] + sh1m)
	v = np.reshape(v,[ind] + sh2m)

	print u.shape, ind1
	print v.shape, ind2

	u = np.swapaxes(u,0,ind1)
	v = np.swapaxes(v,0,ind2)

	print np.sum(np.abs(np.tensordot(u,v,axes=(ind1,ind2))))

	print u.shape, ind1
	print v.shape, ind2

	print '---'

	t1m = Tensor(u.shape,u)
	t2m = Tensor(v.shape,v)

	n1m = n1.modify(t1m, preserveCompressed=True)
	n2m = n2.modify(t2m, preserveCompressed=True)

	newLink = n1m.findLink(n2m)
	newLink.setCompressed()

	return newLink