import numpy as np
from node import Node

def compress(self, tolerance, eps=1e-4):
	n1 = self.__b1.node()
	n2 = self.__b2.node()

	t1 = n1.tensor()
	t2 = n2.tensor()

	arr1 = t1.array()
	arr2 = t2.array()

	ind0 = n1.bucketIndex(self.__b1)
	ind1 = n2.bucketIndex(self.__b2)

	cont = t1.contract(ind0,t2,ind1)
	arrN = cont.array()

	sh1 = list(arr1.shape)
	sh2 = list(arr2.shape)

	sh1m = sh1[:ind0] + sh1[ind0+1:]
	sh2m = sh2[:ind1] + sh2[ind1+1:]

	arrN = np.reshape(arrN,(np.product(sh1m),np.product(sh2m)))

	v, lam, u = np.linalg.svd(arrN,full_matrices=0)

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
	v = np.reshape(v,[ind]+sh1m)
	u = np.reshape(u,[ind]+sh2m)
	v = np.swapaxes(v,0,ind0)
	u = np.swapaxes(u,0,ind1)

	t1m = Tensor(u.shape,u)
	t2m = Tensor(v.shape,v)

	n1m = Node(t1m,n1.network(),children=[n1])
	n2m = Node(t2m,n2.network(),children=[n2])

	n1.setParent(n1m)
	n2.setParent(n2m)

	buckets1 = n1.buckets()
	for i,b in enumerate(buckets1):
		if b.linked():
			nn = b.otherNode(-1)
			bb = b.otherBucket(-1)
			n1m.addLink(nn,i,nn.bucketIndex(bb))
			
	buckets2 = n2.buckets()
	for i,b in enumerate(buckets2):
		if b.linked():
			nn = b.otherNode(-1)
			bb = b.otherBucket(-1)
			n2m.addLink(nn,i,nn.bucketIndex(bb))