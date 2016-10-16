from network import Network
from node import Node
from bucket import Bucket
from link import Link
from arrayTensor import ArrayTensor

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
	indices1 = []
	indices2 = []

	for i in range(len(array.shape)):
		if i in chunkIndices[0]:
			perm.append(c1)
			sh1.append(array.shape[i])
			indices1.append(i)
			c1 += 1
		elif i in chunkIndices[1]:
			perm.append(len(chunkIndices[0]) + c2)
			sh2.append(array.shape[i])
			indices2.append(i)
			c2 += 1
		elif i in ignoreIndex:
			perm.append(len(chunkIndices[0]) + c2)
			sh2.append(array.shape[i])
			indices2.append(i)
			c2 += 1
			prevIndices.append(len(sh2))

	array2 = np.transpose(array, axes=perm)

	array2 = np.reshape(array2, (np.product(sh1),np.product(sh2)))

	u, lam, v = svd(array2, full_matrices=0)

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

	return u,v,prevIndices,indices1,indices2

class treeNetwork(Network):
	'''
	A treeNetwork is a special case of a Network in which the Network being represented
	contains no cycles. This allows matrix elements of a treeNetwork to be efficiently
	evaluated.

	As the only quantities which matter are the matrix elements, the treeNetwork may
	refactor itself through singular value decomposition (SVD) to minimize memory use, and
	so no assumptions should be made about the Nodes in this object, just the external
	Buckets.

	Internally all Nodes of a treeNetwork have rank-3 Tensors. SVD factoring is used
	to enforce this.
	'''

	def __init__(self, accuracy=1e-4):
		'''
		treeNetworks require an accuracy argument which determines how accurately (in terms of relative error)
		they promise to represent their matrix elements.
		'''
		super().__init__()

		self.eps = accuracy

		self.externalBuckets = []

	def pathBetween(self, node1, node2, calledFrom=None):
		'''
		Returns the unique path between node1 and node2.
		This is done by treating node1 as the root of the binary tree and performing a depth-first search.
		'''
		path = []

		if node1 == node2: # Found it!
			return [node1]

		if len(node1.connected()) == 1 and calledFrom is not None:	# Nothing left to search
			return []

		for c in node1.connected(): # Search children
			if c is not calledFrom:
				path2 = self.pathBetween(c, node2, calledFrom=node1)
				if len(path2) > 0:
					return [node1] + path2

		return []

	def splitNode(self, node, prevIndex=None):
		'''
		Takes as input a Node and ensures that it is at most rank 3 by splitting it recursively.
		'''
		if len(node.tensor.shape) > 3:
			self.deregisterNode(node)

			array = node.tensor.array

			if prevIndex is None:
				prevIndex = []

			s = []
			indices = list(range(len(array.shape)))
			for i in prevIndex:
				indices.remove(i)

			for i in indices:
				s.append((i, entropy(array, i)))

			chunks = split_chunks(s, 2)

			chunkIndices = [[i[0] for i in chunks[0]],[i[0] for i in chunks[1]]]

			u, v, prevIndices, indices1, indices2 = splitArray(array, [chunkIndices[0],chunkIndices[1]], ignoreIndex=prevIndex, eps=self.eps)

			b1 = Bucket()
			b2 = Bucket()
			n1 = Node(ArrayTensor(u), self, Buckets=[node.buckets[i] for i in indices1] + [b1])
			n2 = Node(ArrayTensor(v), self, Buckets=[b2] + [node.buckets[i] for i in indices2])
			l = Link(b1,b2)

			chunkIndices[1] = []
			for i in range(len(v.shape)):
				if i != 0 and i not in prevIndices:
					chunkIndices[1].append(i)

			if len(v.shape) <= 3:
				return [n2, self.splitNode(n1, prevIndex=[len(u.shape)-1])]

			self.deregisterNode(n2)
			q, v, prevIndices, indices1, indices2 = splitArray(v, [chunkIndices[1],[0]], ignoreIndex=[0] + prevIndices, eps=self.eps)
			b1 = Bucket()
			b2 = Bucket()
			n3 = Node(ArrayTensor(q), self, Buckets=[n2.buckets[i] for i in indices1] + [b1])
			n4 = Node(ArrayTensor(v), self, Buckets=[b2] + [n2.buckets[i] for i in indices2])
			l = Link(b1,b2)


			return [n4, self.splitNode(n1, prevIndex=[len(u.shape)-1]), self.splitNode(n3, prevIndex=[len(q.shape)-1])]
		else:
			return node
