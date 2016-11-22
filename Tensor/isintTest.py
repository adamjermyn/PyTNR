from bucket import Bucket
from link import Link
from node import Node
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
from network import Network
import numpy as np

nX = 4
nY = 5

nodes = [[] for _ in range(nX)]

n = Network()

x = np.exp(np.random.randn(3,3,3,3))


for i in range(nX):
	for j in range(nY):
		t = AT(x)
		tf = TT()
		tf.addTensor(t)
		nodes[i].append(Node(tf, Buckets=[Bucket() for _ in range(len(x.shape))]))

for i in range(nX):
	for j in range(nY):
		l = Link(nodes[i][j].buckets[0], nodes[(i+1)%nX][j].buckets[2])
		l = Link(nodes[i][j].buckets[1], nodes[i][(j+1)%nY].buckets[3])

for i in range(nX):
	for j in range(nY):
		n.addNode(nodes[i][j])

while len(n.nodes) > 1:
	n1 = next(iter(n.nodes))
	n2 = next(iter(n1.connectedNodes))
	n3 = n.mergeNodes(n1, n2)

	for nn in n3.connectedNodes:
		links = n3.linksConnecting(nn)
		if len(links) > 1:
			buckets1 = [l.bucket1 for l in links if l.bucket1.node is n3]
			buckets2 = [l.bucket2 for l in links if l.bucket2.node is n3]
			buckets = buckets1 + buckets2
			buckets1 = [l.bucket1 for l in links if l.bucket1.node is nn]
			buckets2 = [l.bucket2 for l in links if l.bucket2.node is nn]
			otherBuckets = buckets1 + buckets2
			b3 = n3.mergeBuckets(buckets)
			bn = nn.mergeBuckets(otherBuckets)
			Link(b3, bn)

			assert b3.otherBucket is bn
			assert bn.otherBucket is b3
			assert b3.linked
			assert bn.linked
			assert bn in n.buckets
			assert b3 in n.buckets


	s, s1 = n3.tensor.optimize()
	while s > s1:
		s, s1 = n3.tensor.optimize()
	print(len(n.nodes))
	for nn in n.nodes:
		print(len(nn.tensor.network.nodes)+2, len(nn.tensor.network.externalBuckets))
		print(nn.tensor.network)