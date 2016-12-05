from bucket import Bucket
from node import Node
from link import Link
from network import Network
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
import numpy as np


x = np.random.randn(2,3)
t = AT(x)
tf = TT(1e-5)
tf.addTensor(t)

y = np.random.randn(3,2)
t = AT(y)
tf2 = TT(1e-5)
tf2.addTensor(t)

z = np.random.randn(2,2)
t = AT(z)
tf3 = TT(1e-5)
tf3.addTensor(t)

print('Raw contraction:')
print(np.einsum('ij,jk,ki->',x,y,z))


print('Contraction, order 1:')
tf4 = tf.contract([1],tf2,[0])
tf5 = tf4.contract([0,1],tf3,[1,0])

print(tf5.array)

print('Contraction, order 2:')
tf4 = tf2.contract([1],tf3,[0])
tf5 = tf.contract([0,1],tf4,[1,0])

print(tf5.array)

print('Contraction with nodes, order 1:')
n1 = Node(tf)
n2 = Node(tf2)
n3 = Node(tf3)
Link(n1.buckets[1], n2.buckets[0])
Link(n2.buckets[1], n3.buckets[0])
Link(n3.buckets[1], n1.buckets[0])

nn = Network()
nn.addNode(n1)
nn.addNode(n2)
nn.addNode(n3)

n4 = nn.mergeNodes(n1, n2)
n5 = nn.mergeNodes(n4, n3)
print(n5.tensor.array)

print('Contraction with nodes, order 2:')
n1 = Node(tf)
n2 = Node(tf2)
n3 = Node(tf3)
Link(n1.buckets[1], n2.buckets[0])
Link(n2.buckets[1], n3.buckets[0])
Link(n3.buckets[1], n1.buckets[0])

nn = Network()
nn.addNode(n1)
nn.addNode(n2)
nn.addNode(n3)

n4 = nn.mergeNodes(n1, n3)
n5 = nn.mergeNodes(n4, n2)
print(n5.tensor.array)

print('Contraction with nodes, order 3:')
n1 = Node(tf)
n2 = Node(tf2)
n3 = Node(tf3)
Link(n1.buckets[1], n2.buckets[0])
Link(n2.buckets[1], n3.buckets[0])
Link(n3.buckets[1], n1.buckets[0])

nn = Network()
nn.addNode(n1)
nn.addNode(n2)
nn.addNode(n3)

n4 = nn.mergeNodes(n2, n3)
n5 = nn.mergeNodes(n4, n1)
print(n5.tensor.array)