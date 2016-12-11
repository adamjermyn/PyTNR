from bucket import Bucket
from node import Node
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
import numpy as np


x = np.random.randn(2,3,4,5,6)
t = AT(x)
tf = TT(1e-5)
tf.addTensor(t)

print(tf.shape)
print(x - tf.array)
print(tf.network)

print('-------')

x = np.random.randn(2,3,2,3,2)
t = AT(x)
tf = TT(1e-5)
tf.addTensor(t)

print(tf)

xx = np.random.randn(2,2,2,3,2)
t = AT(xx)
tf2 = TT(1e-5)
tf2.addTensor(t)

print(tf2)

print('-------')

tf3 = tf.contract([0,1,2],tf2,[2,3,4])

print('h',tf3)

print(np.einsum('ijrwp,sqijr->wpsq',x,xx) - tf3.array)