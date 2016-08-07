import tensors
import numpy as np
import networkx

network = tensors.TensorNetwork()

x = np.random.randn(8,8,8,8)
y = np.random.randn(8,8,8,8)
z = np.random.randn(8,8,8,8)
p = np.random.randn(8,8,8,8)

x = network.addTensor(x)
y = network.addTensor(y)
z = network.addTensor(z)
p = network.addTensor(p)

x.addLink(y, 0, 0)
x.addLink(p, 1, 1)
x.addLink(x, 2, 3)

y.addLink(y, 1, 2)
y.addLink(z, 3, 0)

z.addLink(p, 1, 2)

z.addLink(p, 3, 3)

xx = np.random.randn(8,8,8,8)
yy = np.random.randn(8,8,8,8)
zz = np.random.randn(8,8,8,8)
pp = np.random.randn(8,8,8,8)

xx = network.addTensor(xx)
yy = network.addTensor(yy)
zz = network.addTensor(zz)
pp = network.addTensor(pp)

xx.addLink(yy, 0, 0)
xx.addLink(pp, 1, 1)
xx.addLink(xx, 2, 3)

yy.addLink(yy, 1, 2)
yy.addLink(zz, 3, 0)

zz.addLink(pp, 1, 2)
zz.addLink(pp, 3, 3)
pp.addLink(p, 0, 0)

print(network)

for t in network.tensors:
	t.mergeAllLinks()

while len(network.tensors) > 1:
	network = network.merge()
	network.compress()
	network.split()

print(list(network.tensors)[0].array)