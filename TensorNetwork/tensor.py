import numpy as np

class Tensor:
	'''
	A tensor is a multilinear function mapping a series of vectors (also known as indices) to a scalar. 

	Tensors have the following important functions:
	
	shape 		-	This returns a tuple of the same dimension (rank) as the tensor. All entries must be
					integers. The entries may be thought of as specifying the length of
					a multidimensional array along a given axis.
	contract	-	This is a function which takes as input an index, another tensor, and another index,
					and returns a new tensor corresponding to the contraction of this tensor along the
					first index and the other tensor along the second index. This method checks that the
					two tensors are of the same shape along the specified axes and throws a ValueError if
					they are not. The shape of the returned tensor is the concatenation of the shape
					of this tensor with the shape of the other, with the two specified indices removed.
	trace		-	Takes as input two indices with the same size and returns a Tensor which is the result
					of tracing over those indices.
	'''

	def __init__(self, shape, tens):
		self.__shape = shape

		self.__array = np.copy(tens)

	def shape(self):
		return tuple(self.__shape)

	def array(self):
		return self.__array

	def tostr(self):
		return 'Tensor of shape '+str(self.shape())+'.'

	def contract(self, ind, other, otherInd):
		if self.__shape[ind] != other.__shape[otherInd]:
			raise ValueError
		arr = np.dot(self.array(),other.array(),axes=((ind,otherInd)))
		sh1 = list(self.shape())
		sh2 = list(other.shape())
		sh = sh1[:ind] + sh1[ind+1:] + sh2[:otherInd] + sh2[otherInd+1:]
		sh = tuple(sh)
		return Tensor(sh,arr)

	def trace(self, ind0, ind1):
		if self.__shape[ind0] != self.__shape[ind1]:
			raise ValueError
		elif self.array is not None:
			arr = np.trace(self.array(),axis1=ind0,axis2=ind1)
			i0 = min(ind0,ind1)
			i1 = max(ind0,ind1)
			sh = list(self.shape())
			sh = sh[:i0] + sh[i0+1:i1] + sh[i1+1:]
			return Tensor(sh,arr)