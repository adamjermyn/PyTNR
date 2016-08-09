import sys
sys.path.append('../TensorNetwork/')
from network import Network
import numpy as np

### Initialize random arrays

x = np.random.randn(3,3,3,3)
y = np.random.randn(3,3,3,3)
z = np.random.randn(3,3,3,3)
p = np.random.randn(3,3,3,3)

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

x.trace()
y.trace()
z.trace()
p.trace()

### Merge tensors

while len(n.topLevelLinks()) > 0:
	n.merge()

a = n.topLevelNodes()

print list(a)[0].tensor().array()
print arr