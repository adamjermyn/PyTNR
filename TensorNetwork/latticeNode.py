from node import Node
from bucket import Bucket
from tensor import Tensor
import numpy as np

class latticeNode(Node):

	def __init__(self, length, network):
		tens = Tensor(tuple(),np.array(1))
		Node.__init__(self,tens,network,Buckets=[])
		self.__length = length
		self.__dim = 0

	def dim(self):
		return self.__dim

	def addDim(self):
		# TODO: Should probably implement self-factoring so that memory usage doesn't become an issue.

		self.__dim += 1

		if self.__dim > 1:
			arr = np.zeros(tuple(self.__length for i in range(self.__dim)))
			np.fill_diagonal(arr,1.0)
		else:
			arr = np.ones(self.__length)
		tens = Tensor(arr.shape, arr)
		self._Node__tensor = tens

		if self._Node__parent is not None:
			self._Node__parent.delete()

		self._Node__buckets.append(Bucket(self._Node__network))
		self._Node__buckets[-1].addNode(self)

	def removeDim(self):
		assert self.__dim > 0
		self.__dim -= 1

		if self.__dim > 1:
			arr = np.zeros(tuple(self.__length for i in range(self.__dim)))
			np.fill_diagonal(arr,1.0)
		else:
			arr = np.ones(self.__length)
		tens = Tensor(arr.shape, arr)
		self._Node__tensor = tens

		if self._Node__parent is not None:
			self._Node__parent.delete()

		self._Node__buckets[-1].removeNode()
		self._Node__buckets = self._Node__buckets[:-1]



	def addLink(self, other, otherIndex):
		self.addDim()

		if hasattr(other, 'addDim'):
			other.addDim()
			return Node.addLink(self, other, self.dim()-1,other.dim()-1)
		else:
			return Node.addLink(self, other, self.dim()-1,otherIndex)
