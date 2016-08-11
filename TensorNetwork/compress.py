from tensor import Tensor
import numpy as np

def compress(link, eps=1e-4):
	n1 = link.bucket1().node()
	n2 = link.bucket2().node()

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

	u, lam, v = np.linalg.svd(arrN,full_matrices=0)

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	ind = np.searchsorted(cp, eps, side='left')
	ind = len(cp) - ind

	if ind == len(cp):
		link.setCompressed()
		return link

	u = np.transpose(u)

	lam = lam[:ind]
	u = u[:ind,:]
	v = v[:ind,:]

	u *= np.sqrt(lam[:,np.newaxis])
	v *= np.sqrt(lam[:,np.newaxis])

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

	n1m = n1.modify(t1m, preserveCompressed=True)
	n2m = n2.modify(t2m, preserveCompressed=True)
	
	newLink = n1m.findLink(n2m)
	newLink.setCompressed()

	# Remove bad Link. This will actually need to propagate through all children of n1 and n2
	# once the rest of the link inheritance code is written.
	badLink = n1m.findLink(n2)
	badLink.delete()
	badLink = n2m.findLink(n1)
	badLink.delete()

	return newLink, n1m, n2m