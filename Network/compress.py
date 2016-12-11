import numpy as np
from utils import ndArrayToMatrix, generalSVD, matrixToNDArray, matrixProductLinearOperator

def compressLink(l, accuracy):
	b1 = l.bucket1
	b2 = l.bucket2

	ind1 = b1.index
	ind2 = b2.index

	n1 = b1.node
	n2 = b2.node

	a1, ind1I = n1.tensor.getIndexFactor(ind1)
	a2, ind2I = n2.tensor.getIndexFactor(ind2)

	sh1 = list(a1.shape)
	sh2 = list(a2.shape)

	sh1m = sh1[:ind1I] + sh1[ind1I+1:]
	sh2m = sh2[:ind2I] + sh2[ind2I+1:]

	a1 = ndArrayToMatrix(a1, ind1I, front=False)
	a2 = ndArrayToMatrix(a2, ind2I, front=True)

	if a1.shape[1] < a1.shape[0] and a2.shape[0] < a2.shape[1]:
		arr = matrixProductLinearOperator(a1, a2)
		u, lam, v, p, cp = generalSVD(arr, bondDimension=a1.shape[1])
	else:
		arr = np.dot(a1, a2)
		u, lam, v, p, cp = generalSVD(arr)

	ind = np.searchsorted(cp, accuracy, side='left')
	ind = len(cp) - ind

	u = u[:,:ind]
	lam = lam[:ind]
	v = v[:ind,:]

	u *= np.sqrt(lam)[np.newaxis,:]
	v *= np.sqrt(lam)[:,np.newaxis]

	u = matrixToNDArray(u, sh1[:ind1I] + [ind] + sh1[ind1I+1:], ind1I, front=True)
	v = matrixToNDArray(v, sh2[:ind2I] + [ind] + sh2[ind2I+1:], ind2I, front=False)

	n1.tensor = n1.tensor.setIndexFactor(ind1, u)
	n2.tensor = n2.tensor.setIndexFactor(ind2, v)

	assert n1.tensor.shape[ind1] == n2.tensor.shape[ind2]