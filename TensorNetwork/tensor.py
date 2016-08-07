import numpy as np

class Tensor:
	'''
	A tensor is a multilinear function mapping a series of vectors (also known as indices) to a scalar. 

	Tensors have the following functions:
	
	shape 		-	This returns a tuple of the same dimension (rank) as the tensor. All entries must be
					either integers or +infinity (corresponding to the case of a continuum tensor).
					In the case of integers, the entries may be thought of as specifying the length of
					a multidimensional array along a given axis. The infinite case may be used when
					the tensor is a function of a continuous variable.
	contract	-	This is a function which takes as input an index, another tensor, and another index,
					and returns a new tensor corresponding to the contraction of this tensor along the
					first index and the other tensor along the second index. This method checks that the
					two tensors are of the same shape along the specified axes and throws a ValueError if
					they are not. The shape of the returned tensor is the concatenation of the shape
					of this tensor with the shape of the other, with the two specified indices removed.

	Currently the contract method will raise a NotImplementedError if given a continuum function.
	'''

	def __init__(self, shape, tens):
		self.__shape = shape

		self.__function = None
		self.__array = None

		# tens can either be a function or an array.
		# If it is an array, it must be one with shape of shape.
		if hasattr(tens,'__call__'):
			self.function = tens
		else:
			self.array = np.copy(tens)

	def shape(self):
		return tuple(self.__shape)

	def tostr(self):
		return 'Tensor of shape '+str(self.shape())+'.'

	def contract(self, ind, other, otherInd):
		if self.shape()[ind] != other.shape()[otherInd]:
			raise ValueError
		elif self.array is not None:
			arr = np.dot(self.array,other.array,axes=((ind,otherInd)))
			sh1 = list(self.shape())
			sh2 = list(other.shape())
			sh = sh1[:ind] + sh1[ind+1:] + sh2[:otherInd] + sh2[otherInd+1:]
			sh = tuple(sh)
			return Tensor(sh,arr)
		else:
			raise NotImplementedError