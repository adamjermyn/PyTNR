import numpy as np
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from utils import ndArrayToMatrix, matrixToNDArray

tempdir = tempfile.TemporaryDirectory()
executor = ThreadPoolExecutor(max_workers=10)
maxSize = 500000

def write(fname, arr):
	fi = open(fname,'wb+')
	np.save(fi,arr,allow_pickle=False)
	fi.close()

def read(fname):
	fi = open(fname,'rb')
	arr = np.load(fi)
	fi.close()
	return arr	

class Tensor:
	'''
	A tensor is a multilinear function mapping a series of vectors (also known as indices) to a scalar. 
	'''

	def __init__(self, shape, tens):
		assert shape == tens.shape

		self.__shape = shape
		self.__size = tens.size

		# We normalize the Tensor by factoring out the log of the
		# maximum-magnitude element.
		m = np.max(np.abs(tens))
		self.__logScalar = np.log(m)

		if self.__size < maxSize:
			self.__array = np.copy(tens/m)
		else:
			handle, self.__array = tempfile.mkstemp(dir=tempdir.name)
			os.close(handle)
			self.__writeFuture = executor.submit(write,self.__array, tens/m)

	def shape(self):
		'''
		Returns the shape of the Tensor.
		'''
		return tuple(self.__shape)

	def size(self):
		'''
		Returns the size of the Tensor (the number of elements stored)
		'''
		return self.__size

	def array(self):
		'''
		Returns the array underlying the Tensor.
		Also handles any caching operations that are needed for large-memory Tensors.
		'''
		if self.__size < maxSize:
			return np.copy(self.__array)
		else:
			self.__writeFuture.result()
			return read(self.__array)

	def logScalar(self):
		'''
		Returns the log-scalar component of the Tensor.
		The exponential of this is multiplied by the Tensor's array to recover
		the full Tensor array.
		'''	
		return self.__logScalar

	def tostr(self):
		return 'Tensor of shape '+str(self.shape())+'.'

	def contract(self, ind, other, otherInd):
		'''
		Takes as input:
			ind 		-	A list of indices on this Tensor.
			other 		-	The other Tensor.
			otherInd	-	A list of indices on the other Tensor.

		Returns a Tensor containing the contraction of this Tensor with the other.
		'''
		arr = np.tensordot(self.array(),other.array(),axes=((ind,otherInd)))
		return Tensor(arr.shape,arr)

	def trace(self, ind0, ind1):
		'''
		Takes as input:
			ind0	-	A list of indices on one side of their Links.
			ind1	-	A list of indices on the other side of their Links.

		Elements of ind0 and ind1 must correspond, such that the same Link is
		represented by indices at the same location in each list.

		Elements of ind0 should not appear in ind1, and vice-versa.

		Returns a Tensor containing the trace over all of the pairs of indices.
		'''
		arr = self.array()
		ind0 = list(ind0)
		ind1 = list(ind1)

		for i in range(len(ind0)):
			arr = np.trace(arr, axis1=ind0[i], axis2=ind1[i])
			for j in range(len(ind0)):
				d0 = 0
				d1 = 0

				if ind0[j] > ind0[i]:
					d0 += 1
				if ind0[j] > ind1[i]:
					d0 += 1
	
				if ind1[j] > ind0[i]:
					d1 += 1
				if ind1[j] > ind1[i]:
					d1 += 1

				ind0[j] -= d0
				ind1[j] -= d1

		return Tensor(arr.shape,arr)




#######################
# Tensor Helper Methods
#######################

def tensorToMatrix(tens, index, front=True):
		'''
		This method flattens the Tensor's array along all indices other than
		index and does so in a way which preserves the ordering of the other
		axes when unflattened.

		This method also takes as input a boolean variable front. If front is True
		then the special index is pushed to the beginning. If front is False then the
		special index is pushed to the back.
		'''
		return ndArrayToMatrix(tens.array(), index, front=front)

def matrixToTensor(matrix, shape, index, front=True):
		'''
		This method takes a 2D array and reshapes it to the given shape.
		The reshape operation only modifies one of the axes of the matrix.
		This is either the first (front) or last (not front) depending on the
		boolean variable front. Whichever index is not reshaped is then
		put in the position specified by index.

		This method is meant to be the inverse of tensorToMatrix.
		'''
		arr = matrixToNDArray(matrix, shape, index, front=front)
		return Tensor(arr.shape, arr)
