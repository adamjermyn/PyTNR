from bucket import Bucket
from link import Link
from node import Node
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
from network import Network
import numpy as np
from compress import compressLink

nX = 3
nY = 3
nZ = 3
accuracy = 1e-4

nodes = [[[] for _ in range(nY)] for _ in range(nX)]

n = Network()

x = np.exp(np.random.randn(2,2,2,2,2,2))


for i in range(nX):
	for j in range(nY):
		for k in range(nZ):
			t = AT(x)
			tf = TT(accuracy)
			tf.addTensor(t)
			nodes[i][j].append(Node(tf, Buckets=[Bucket() for _ in range(len(x.shape))]))

for i in range(nX):
	for j in range(nY):
		for k in range(nZ):
			l = Link(nodes[i][j][k].buckets[0], nodes[(i+1)%nX][j][k].buckets[3])
			l = Link(nodes[i][j][k].buckets[1], nodes[i][(j+1)%nY][k].buckets[4])
			l = Link(nodes[i][j][k].buckets[2], nodes[i][j][(k+1)%nZ].buckets[5])

for i in range(nX):
	for j in range(nY):
		for k in range(nZ):
			n.addNode(nodes[i][j][k])

while len(n.nodes) > 1:
	smallest = [1e20,None,None]
	for nn in n.nodes:
		for nnn in nn.connectedNodes:
			commonNodes = set(nn.connectedNodes).intersection(nnn.connectedNodes)
			metric = nn.tensor.rank + nnn.tensor.rank - len(commonNodes)
			if metric < smallest[0]:
				smallest[0] = metric
				smallest[1] = nn
				smallest[2] = nnn

	n1 = smallest[1]
	n2 = smallest[2]
	n3 = n.mergeNodes(n1, n2)
	n.mergeLinks(n3, accuracy=accuracy)
	n3.tensor.optimize()

	for nn in n.nodes:
		print(nn.tensor.shape, nn.tensor.size, 1.0*nn.tensor.size/np.product(nn.tensor.shape))

	print('-------------------------------')
	print('-------',smallest[0],len(n3.connectedNodes),len(n.nodes),'-------')
	print('-------------------------------')
