import tensors
import numpy as np

x = np.random.randn(3,3,3,3)
y = np.random.randn(3,3,3,3)
z = np.random.randn(3,3,3,3)
p = np.random.randn(3,3,3,3)

network = tensors.TensorNetwork()

arr = np.einsum('abcc,aeeg,gdfh,qbdh->fq',x,y,z,p)

x = network.addTensor(x)
y = network.addTensor(y)
z = network.addTensor(z)
p = network.addTensor(p)

x.addLink(y, 0, 0)
x.addLink(p, 1, 1)
x.addLink(x, 2, 3)

print(network)

y.addLink(y, 1, 2)
y.addLink(z, 3, 0)

print(network)

z.addLink(p, 1, 2)

print(network)

z.addLink(p, 3, 3)

print(network)

print('-----')

while len(network.all_links) > 0:
	l = list(network.all_links)[0]
	print(network)
	network.mergeNodes(l)

print(list(network.tensors)[0])
print(list(network.tensors)[0].array)
print(arr)
