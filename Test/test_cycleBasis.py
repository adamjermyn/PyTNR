import numpy as np

from TNRG.Network.network import Network
from TNRG.Network.node import Node
from TNRG.Network.link import Link
from TNRG.Network.cycleBasis import cycleBasis
from TNRG.Network.treeNetwork import TreeNetwork
from TNRG.Tensor.arrayTensor import ArrayTensor

def test_init():
	net = Network()

	nodes = []
	for i in range(3):
		x = np.random.randn(2,2,2)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		net.addNode(n1)
		nodes.append(n1)

	l1 = Link(nodes[0].buckets[0],nodes[1].buckets[0])
	l2 = Link(nodes[1].buckets[1],nodes[2].buckets[0])
	l3 = Link(nodes[2].buckets[1],nodes[0].buckets[1])

	cb = cycleBasis(net)

	assert len(cb.cycles) == 1
	assert len(cb.cycles[0]) == 3
	assert l1 in cb.cycles[0]
	assert l2 in cb.cycles[0]
	assert l3 in cb.cycles[0]

def test_triangle():
	net = Network()

	nodes = []
	for i in range(3):
		x = np.random.randn(2,2,2)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		net.addNode(n1)
		nodes.append(n1)

	l1 = Link(nodes[0].buckets[0],nodes[1].buckets[0])
	l2 = Link(nodes[1].buckets[1],nodes[2].buckets[0])
	l3 = Link(nodes[2].buckets[1],nodes[0].buckets[1])

	cb = cycleBasis(net)

	cycle = cb.smallest()
	cb.mergeSmall(cycle)

	print(cb.cycles)

	assert len(cb.cycles) == 0

def test_two_triangle():
	net = Network()

	nodes = []
	for i in range(4):
		x = np.random.randn(2,2,2)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		net.addNode(n1)
		nodes.append(n1)

	l1 = Link(nodes[0].buckets[0],nodes[1].buckets[0])
	l2 = Link(nodes[1].buckets[1],nodes[2].buckets[0])
	l3 = Link(nodes[2].buckets[1],nodes[0].buckets[1])
	l4 = Link(nodes[0].buckets[2],nodes[3].buckets[0])
	l5 = Link(nodes[1].buckets[2],nodes[3].buckets[1])

	cb = cycleBasis(net)

	assert len(cb.cycles) == 2

	cycle = cb.smallest()
	print('aldjaslkdjakljsdjasdilasj',len(cycle))
	cb.mergeSmall(cycle)

	assert len(cb.cycles) == 1

	cycle = cb.smallest()
	cb.mergeSmall(cycle)

	assert len(cb.cycles) == 0

def test_swap():
	net = TreeNetwork()

	nodes = []
	for i in range(4):
		x = np.random.randn(2,2,2)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		net.addNode(n1)
		nodes.append(n1)

	l1 = Link(nodes[0].buckets[0],nodes[1].buckets[0])
	l2 = Link(nodes[1].buckets[1],nodes[2].buckets[0])
	l3 = Link(nodes[2].buckets[1],nodes[0].buckets[1])
	l4 = Link(nodes[0].buckets[2],nodes[3].buckets[0])
	l5 = Link(nodes[1].buckets[2],nodes[3].buckets[1])

	cb = cycleBasis(net)

	assert len(cb.cycles) == 2

	cb.swap(l1, nodes[0].buckets[2],nodes[1].buckets[1])

	assert len(cb.cycles) == 2

	assert len(cb.cycles[0]) == 3
	assert len(cb.cycles[1]) == 3

	new = list(cb.cycles[0])
	if l1 in new:
		new.remove(l1)
	if l2 in new:
		new.remove(l2)
	if l3 in new:
		new.remove(l3)
	if l4 in new:
		new.remove(l4)
	if l5 in new:
		new.remove(l5)

	assert len(new) == 1

	new = new.pop()

	assert new in cb.cycles[1]