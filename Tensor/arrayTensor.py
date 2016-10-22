import numpy as np
from tensor import Tensor

class ArrayTensor(Tensor):

	def __init__(self, tens):
		self._shape = tens.shape
		self._rank = len(self._shape)

		self._size = tens.size

		# We normalize the Tensor by factoring out the log of the
		# maximum-magnitude element.
		m = np.max(np.abs(tens))
		self._logScalar = np.log(m)
		self.array = np.copy(tens/m)

	def __str__(self):
		return 'Tensor of shape '+str(self.shape)+'.'

	@property
	def shape(self):
		return self._shape

	@property
	def rank(self):
		return self._rank

	@property
	def size(self):
		return self._size

	@property
	def logScalar(self):
		return self._logScalar

	def contract(self, ind, other, otherInd):
		'''
		Takes as input:
			ind 		-	A list of indices on this Tensor.
			other 		-	The other Tensor.
			otherInd	-	A list of indices on the other Tensor.

		Returns a Tensor containing the contraction of this Tensor with the other.
		'''
		print(self.shape,other.shape)
		arr = np.tensordot(self.array,other.array,axes=((ind,otherInd)))
		return Tensor(arr)

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
		arr = self.array
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

		return ArrayTensor(arr)

	def __deepcopy__(self, memo):
		return self
