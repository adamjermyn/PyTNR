from node import Node
from bucket import Bucket
from tensor import Tensor
import numpy as np
from utils import kroneckerDelta

class latticeNode(Node):

	def __init__(self, length, network):
		tens = Tensor(tuple(),np.array(1))
		Node.__init__(self,tens,network,Buckets=[])
		self._length = length
		self._dim = 0

	def dim(self):
		return self._dim

	def addDim(self):
		# TODO: Should probably implement self-factoring so that memory usage doesn't become an issue.

		self._dim += 1

		arr = kroneckerDelta(self._dim, self._length)
		tens = Tensor(arr.shape, arr)
		self._tensor = tens

		if self._parent is not None:
			self._parent.delete()

		self._buckets.append(Bucket(self._network))
		self._buckets[-1].addNode(self)

	def removeDim(self):
		assert self._dim > 0
		self._dim -= 1

		arr = kroneckerDelta(self._dim, self._length)
		tens = Tensor(arr.shape, arr)
		self._tensor = tens

		if self._parent is not None:
			self._parent.delete()

		self._buckets[-1].removeNode()
		self._buckets = self._buckets[:-1]



	def addLink(self, other, otherIndex):
		self.addDim()

		if hasattr(other, 'addDim'):
			other.addDim()
			return Node.addLink(self, other, self.dim()-1,other.dim()-1)
		else:
			return Node.addLink(self, other, self.dim()-1,otherIndex)
