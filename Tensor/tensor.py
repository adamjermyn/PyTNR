from abc import ABC, abstractmethod

class Tensor(ABC):
	'''
	A tensor is a multilinear function mapping a series of vectors (also known as indices) to a scalar.
	It must have shape and size attributes and an attribute returning the logarithm of the largest-magnitude element.
	It must implement __str__ (for printing), as well as contraction with another tensor and tracing between
	a pair of its indices.
	'''

	@abstractmethod
	def __str__(self):
		'''
		Returns a string representation of the Tensor.
		'''
		pass	

	@property
	@abstractmethod
	def shape(self):
		'''
		Returns the shape of the Tensor as a tuple.
		'''
		pass	

	@property
	@abstractmethod
	def size(self):
		'''
		Returns the number of elements of the Tensor.
		'''
		pass	

	@property
	@abstractmethod
	def logScalar(self):
		'''
		Returns the logarithm of the largest-magnitude element in the Tensor.
		'''
		pass
			
	@abstractmethod
	def contract(self, ind, other, otherInd):
		'''
		Returns a Tensor representing the contraction of self at indices ind with other at indices otherInd,
		where ind and otherInd may be either integers or lists of integers. If they are lists they must
		have the same shape.
		'''
		pass

	@abstractmethod
	def trace(self, ind0, ind1):
		'''
		Returns the trace of the Tensor along the pairs of indices specified by ind0 and ind1, where ind0 and ind1
		are either integers or lists of integers of the same shape.
		'''
		pass