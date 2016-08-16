from node imoprt Node
from tensor import Tensor

class latticeNode(Node):

	def __init__(self, length, network):
		tens = Tensor(tuple(),np.array(1))
		Node.__init__(tens,network,Buckets=[Bucket()])
		self.__length = length
		self.__dim = 9

	def dim():
		return self.__dim

	def addDim(self):
		self.__dim += 1

		if self.__dim > 1:
			arr = np.zeros((self.__length for i in range(self.__dim)))
			np.fill_diagonal(arr,1.0)
		else:
			arr = np.ones(self.__length)
		tens = Tensor(arr.shape, arr)

		self.__tensor = tens
		self.__buckets.append(Bucket())

		# Should also delete all ancestors

	def addLink(self, other):
		# self and other must both be latticeNodes
		self.addDim()
		other.addDim()

		return Node.addLink(self, other, self.dim()-1,other.dim()-1)
