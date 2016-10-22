from bucket import Bucket
from node import Node
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
import numpy as np


x = np.random.randn(5,5,5,5,5,5)
t = AT(x)
net = TN()
n = Node(t, Buckets=[Bucket() for _ in range(t.rank)])
net.addNode(n)
net.splitNode(n)
tf = TT(net)

print(tf)

x = np.random.randn(5,5,5,5,5,5)
t = AT(x)
net = TN()
n = Node(t, Buckets=[Bucket() for _ in range(t.rank)])
net.addNode(n)
net.splitNode(n)
tf2 = TT(net)

print(tf2)

tf3 = tf.contract([0,1,2,3,4],tf2,[0,1,2,3,4])

print(tf3)
