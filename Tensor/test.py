from bucket import Bucket
from node import Node
from treeTensor import TreeTensor as TT
from arrayTensor import ArrayTensor as AT
from treeNetwork import TreeNetwork as TN
import numpy as np


x = np.random.randn(4,4,4,4,4)
t = AT(x)
tf = TT(1e-5)
tf.addTensor(t)

print(tf)

xx = np.random.randn(4,4,4,4,4)
t = AT(xx)
tf2 = TT(1e-5)
tf2.addTensor(t)

print(tf2)

print('-------')

tf3 = tf.contract([0,1,2],tf2,[2,3,4])

print(tf3)

print(tf3.network.array())
print(np.einsum('ijrwp,sqijr->wspq',x,xx))
print(np.sum(tf3.network.array()**2))
print(np.sum(np.einsum('ijrwp,sqijr->wspq',x,xx)**2))

tf4 = tf3.trace(0,1)
