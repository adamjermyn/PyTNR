import sys
sys.path.append('../TensorNetwork/')
from network import Network
import numpy as np

### Initialize random arrays

x = np.random.randn(2,2,2,2)
y = np.random.randn(2,2,2,2)
z = np.random.randn(2,2,2,2)
p = np.random.randn(2,2,2,2)

x = np.exp(x)
y = np.exp(y)
z = np.exp(z)
p = np.exp(p)

x = np.abs(x)
y = np.abs(y)
z = np.abs(z)
p = np.abs(p)

x = np.ones((2,2,2,2))
y = np.ones((2,2,2,2))
z = np.ones((2,2,2,2))
p = np.ones((2,2,2,2))
p[1,1,1,1] = 2.0
z[1,1,0,1] = 2.0

### Compute final answer

arr = np.einsum('abcc,aeeg,gdfh,qbdh->fq',x,y,z,p)

### Initialize network

n = Network()

### Add tensors

x = n.addNodeFromArray(x)
y = n.addNodeFromArray(y)
z = n.addNodeFromArray(z)
p = n.addNodeFromArray(p)

### Link tensors

x.addLink(y, 0, 0)
x.addLink(p, 1, 1)
x.addLink(x, 2, 3)

y.addLink(y, 1, 2)
y.addLink(z, 3, 0)

z.addLink(p, 1, 2)

z.addLink(p, 3, 3)

### Trace

n.contract(mergeL=True, compressL=True, eps=1e-10)

arrT, logS, buckets = n.topLevelRepresentation()

arrT *= np.exp(logS)

print('Error is the smaller of the following two:')
print(np.sum(np.abs(arrT-arr))/np.sum(np.abs(arr)))
print(np.sum(np.abs(arrT-np.transpose(arr)))/np.sum(np.abs(arr)))