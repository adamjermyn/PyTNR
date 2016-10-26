from bucket import Bucket
from node import Node
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
import numpy as np


x = np.random.randn(4,4,4)
t = AT(x)
net = TN()
n = Node(t, Buckets=[Bucket() for _ in range(t.rank)])
net.addNode(n)
net.splitNode(n)
tf = TT(net)

print(tf)

xx = np.random.randn(4,4,4)
t = AT(xx)
net = TN()
n = Node(t, Buckets=[Bucket() for _ in range(t.rank)])
net.addNode(n)
net.splitNode(n)
tf2 = TT(net)

print(tf2)

print('-------')

tf3 = tf.contract([0,1],tf2,[0,1])

print(tf3)

print(tf3.network.array())
print(np.einsum('ijp,ijq->pq',x,xx))
print(np.sum(tf3.network.array()**2))
print(np.sum(np.einsum('ijp,ijq->pq',x,xx)**2))