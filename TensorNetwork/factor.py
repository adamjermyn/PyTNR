import numpy as np
from scipy.linalg import logm
from numpy.linalg import svd

def entropy(array, index):
	array2 = np.swapaxes(np.copy(array), 0, index)
	array2 = np.reshape(array2, (array.shape[index],-1))
	array2 = np.dot(array2, np.transpose(np.conjugate(array2)))
	array2 /= np.trace(array2)
	return -np.trace(np.dot(array2,logm(array2)))

def split_chunks(l, n):
    """ 
       Splits list l into n chunks with approximately equals sum of values
       see  http://stackoverflow.com/questions/6855394/splitting-list-in-chunks-of-balanced-weight
    """
    result = [[] for i in range(n)]
    sums   = {i:0 for i in range(n)}
    c = 0
    for j,e in l:
        for i in sums:
            if c == sums[i]:
                result[i].append((j,e))
                break
        sums[i] += e
        c = min(sums.values())    
    return result

def splitArray(array, chunkIndices, ignoreIndex=None, eps=1e-4):
	if ignoreIndex is None:
		ignoreIndex = []


	perm = []
	prevIndices = []

	c1 = 0
	c2 = 0
	sh1 = []
	sh2 = []
	for i in range(len(array.shape)):
		if i in chunkIndices[0]:
			perm.append(c1)
			sh1.append(array.shape[i])
			c1 += 1
		elif i in chunkIndices[1]:
			perm.append(len(chunkIndices[0]) + c2)
			sh2.append(array.shape[i])
			c2 += 1
		elif i in ignoreIndex:
			perm.append(len(chunkIndices[0]) + c2)
			sh2.append(array.shape[i])
			c2 += 1
			prevIndices.append(len(sh2))

	array2 = np.transpose(array, axes=perm)

	array2 = np.reshape(array2, (np.product(sh1),np.product(sh2)))

	u, lam, v = svd(array2)

	p = lam**2
	p /= np.sum(p)
	cp = np.cumsum(p[::-1])

	ind = np.searchsorted(cp, eps, side='left')
	ind = len(cp) - ind

	u = u[:,:ind]
	lam = lam[:ind]
	v = v[:ind,:]

	u *= np.sqrt(lam)[np.newaxis,:]
	v *= np.sqrt(lam)[:,np.newaxis]

	u = np.reshape(u, sh1 + [ind])
	v = np.reshape(v, [ind] + sh2)

	return u,v,prevIndices

def iterativeSplit(array, prevIndex=None, eps=1e-4):
	if len(array.shape) <= 3:
		return [array]
	else:

		if prevIndex is None:
			prevIndex = []

		s = []
		indices = list(range(len(array.shape)))
		for i in prevIndex:
			indices.remove(i)

		for i in indices:
			s.append((i,entropy(array,i)))

		chunks = split_chunks(s, 2)

		chunkIndices = [[i[0] for i in chunks[0]],[i[0] for i in chunks[1]]]

		u, v, prevIndices = splitArray(array, [chunkIndices[0],chunkIndices[1]], ignoreIndex=prevIndex, eps=eps)

		chunkIndices[1] = []
		for i in range(len(v.shape)):
			if i != 0 and i not in prevIndices:
				chunkIndices[1].append(i)

		if len(v.shape) <= 3:
			return [v, iterativeSplit(u, prevIndex=[len(u.shape)-1], eps=eps)]

		q, v, prevIndices = splitArray(v, [chunkIndices[1],[0]], ignoreIndex=[0] + prevIndices, eps=eps)

		return [v, iterativeSplit(u, prevIndex=[len(u.shape)-1], eps=eps), iterativeSplit(q, prevIndex=[len(q.shape)-1], eps=eps)]

def treeIterator(tree):
	ret = []
	ret.append(tree[0].shape)
	for t in tree[1:]:
		ret.append(treeIterator(t))
	return ret

def treeSize(tree):
	ret = 0
	ret += tree[0].size
	for t in tree[1:]:
		ret += treeSize(t)
	return ret

#array = np.random.randn(4,4,4,4,4,4,4)
array = np.einsum('ij,jka,jml,aqw->ikmlqw',np.random.randn(8,8),np.random.randn(8,4,4),np.random.randn(8,3,4),np.random.randn(4,4,4))

split = iterativeSplit(array)
print(treeSize(split)/array.size)
print(treeIterator(split))
