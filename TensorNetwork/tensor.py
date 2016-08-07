class Tensor:
	'''
	A tensor is a multilinear function mapping a series of vectors (also known as indices) to a scalar. 

	Tensors have the following properties:
	
	shape 		-	This is a tuple of the same dimension (rank) as the tensor. All entries must be
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
	'''

	def __init__(self, shape):
		

