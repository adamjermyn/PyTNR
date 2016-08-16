import numpy as np

class Tensor:
	'''
	A tensor is a multilinear function mapping a series of vectors (also known as indices) to a scalar. 

	Tensors have the following important functions:
	
	shape 		-	This returns a tuple of the same dimension (rank) as the tensor. All entries must be
					integers. The entries may be thought of as specifying the length of
					a multidimensional array along a given axis.
	size		-	Returns the number of elements in the array.
	contract	-	This is a function which takes as input an index, another tensor, and another index,
					and returns a new tensor corresponding to the contraction of this tensor along the
					first index and the other tensor along the second index. This method checks that the
					two tensors are of the same shape along the specified axes and throws a ValueError if
					they are not. The shape of the returned tensor is the concatenation of the shape
					of this tensor with the shape of the other, with the two specified indices removed.
	trace		-	Takes as input two indices with the same size and returns a Tensor which is the result
					of tracing over those indices.
	makeUnity 	-	Divides the Tensor through by the quantity needed to make the largest entry +-1.
					Returns the log of this quantity.
	'''

	def __init__(self, shape, tens):
		self.__shape = shape
		self.__array = np.copy(tens)
		self.__size = self.__array.size

	def shape(self):
		return tuple(self.__shape)

	def size(self):
		return self.__size

	def array(self):
		return self.__array

	def tostr(self):
		return 'Tensor of shape '+str(self.shape())+'.'

	def contract(self, ind, other, otherInd):
		arr = np.tensordot(self.array(),other.array(),axes=((ind,otherInd)))
		return Tensor(arr.shape,arr)

	def trace(self, ind0, ind1):
		if self.__shape[ind0] != self.__shape[ind1]:
			raise ValueError
		arr = np.trace(self.array(),axis1=ind0,axis2=ind1)
		return Tensor(arr.shape,arr)

	def makeUnity(self):
		m = np.max(np.abs(self.__array))
		self.__array /= m
		return m

	def deepcopy(self):
		arr = np.copy(self.__array)
		return Tensor(arr.shape, arr)