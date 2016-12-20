import numpy as np

from TNRG.Network.network import Network
from TNRG.Network.node import Node
from TNRG.Network.link import Link
from TNRG.Tensor.arrayTensor import ArrayTensor
from TNRG.TreeTensor.treeTensor import TreeTensor

epsilon = 1e-10

def test_init():
	net = Network()

	assert len(net.nodes) == 0
	assert len(net.buckets) == 0
	assert len(net.internalBuckets) == 0
	assert len(net.externalBuckets) == 0
	assert len(net.optimizedLinks) == 0

	x = np.random.randn(2,3,3)
	xt = ArrayTensor(x)
	n1 = Node(xt)

	net.addNode(n1)

	assert len(net.nodes) == 1
	assert len(net.buckets) == 3
	assert len(net.internalBuckets) == 0
	assert len(net.externalBuckets) == 3
	assert len(net.optimizedLinks) == 0

	x = np.random.randn(2,3,3)
	xt = ArrayTensor(x)
	n2 = Node(xt)
	Link(n1.buckets[0], n2.buckets[0])

	net.addNode(n2)

	assert len(net.nodes) == 2
	assert len(net.buckets) == 6
	assert len(net.internalBuckets) == 2
	assert len(net.externalBuckets) == 4
	assert len(net.optimizedLinks) == 0

	net.removeNode(n1)

	assert len(net.nodes) == 1
	assert len(net.buckets) == 3
	assert len(net.internalBuckets) == 0
	assert len(net.externalBuckets) == 3
	assert len(net.optimizedLinks) == 0

def test_mergeNode_ArrayTensor():
	for i in range(5):
		net = Network()

		x = np.random.randn(2,3,3)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		xt = ArrayTensor(x)
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[0], n2.buckets[0])
		net.addNode(n2)

		net.mergeNodes(n1, n2)

		assert len(net.nodes) == 1
		assert len(net.buckets) == 4
		assert len(net.internalBuckets) == 0
		assert len(net.externalBuckets) == 4
		assert len(net.optimizedLinks) == 0

		arr, bdict = net.array
		assert arr.shape == (3,3,3,3)
		for b in net.buckets:
			assert b.id in bdict
		for b1 in net.buckets:
			for b2 in net.buckets:
				if b1.id < b2.id:
					assert bdict[b1.id] < bdict[b2.id]

		assert np.sum((arr - np.einsum('ijk,ilm->jklm',x,x))**2) < epsilon

def test_mergeNode_TreeTensor():
	for i in range(5):
		net = Network()

		x = np.random.randn(2,2,2,2,2)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n1 = Node(xt)
		y = np.random.randn(2,2,2,2,2)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(y))
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[0], n2.buckets[0])
		net.addNode(n2)

		net.mergeNodes(n1, n2)

		assert len(net.nodes) == 1
		assert len(net.buckets) == 8
		assert len(net.internalBuckets) == 0
		assert len(net.externalBuckets) == 8
		assert len(net.optimizedLinks) == 0

		arr, bdict = net.array
		assert arr.shape == (2,2,2,2,2,2,2,2)
		for b in net.buckets:
			assert b.id in bdict
		for b1 in net.buckets:
			for b2 in net.buckets:
				if b1.id < b2.id:
					assert bdict[b1.id] < bdict[b2.id]

		assert np.sum((arr - np.einsum('ijklm,iqwer->jklmqwer',x,y))**2) < epsilon

def test_mergeLinks_ArrayTensor():
	for i in range(5):
		net = Network()

		x = np.random.randn(2,2,3,3)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		xt = ArrayTensor(x)
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[0], n2.buckets[0])
		Link(n1.buckets[1], n2.buckets[1])
		net.addNode(n2)

		arr1, bdict1 = net.array

		net.mergeLinks(n1)

		arr2, bdict2 = net.array

		assert np.sum((arr1 - arr2)**2) < epsilon
		assert np.sum((arr1 - arr2)**2) < epsilon

def test_mergeLinks_TreeTensor():
	for i in range(10):
		net = Network()

		x = np.random.randn(2,2,2,2,2)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n1 = Node(xt)
		x = np.random.randn(2,2,2,2,2)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[0], n2.buckets[0])
		Link(n1.buckets[1], n2.buckets[1])
		net.addNode(n2)

		arr1, bdict1 = net.array

		net.mergeLinks(n1)

		arr2, bdict2 = net.array

		assert np.sum((arr1 - arr2)**2) < epsilon

	for i in range(5):
		net = Network()

		x = np.random.randn(2,2,2,2,3,3)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n1 = Node(xt)
		x = np.random.randn(2,2,2,2,3,3)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[4], n2.buckets[5])
		Link(n1.buckets[5], n2.buckets[4])
		net.addNode(n2)

		arr1, bdict1 = net.array

		net.mergeLinks(n1)

		arr2, bdict2 = net.array

		assert np.sum((arr1 - arr2)**2) < epsilon

	for i in range(5):
		net = Network()

		x = np.random.randn(2,2,2,2,3,3)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n1 = Node(xt)
		x = np.random.randn(2,2,2,2,3,3)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[3], n2.buckets[2])
		Link(n1.buckets[4], n2.buckets[4])
		net.addNode(n2)

		arr1, bdict1 = net.array

		net.mergeLinks(n1)

		arr2, bdict2 = net.array

		assert np.sum((arr1 - arr2)**2) < epsilon


def test_mergeLinks_ArrayTensor_Compress():
	for i in range(5):
		net = Network()

		x = np.random.randn(2,2,2,2)
		xt = ArrayTensor(x)
		n1 = Node(xt)
		xt = ArrayTensor(x)
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[0], n2.buckets[0])
		Link(n1.buckets[1], n2.buckets[1])
		net.addNode(n2)

		arr1, bdict1 = net.array

		net.mergeLinks(n1, compress=True, accuracy=epsilon)

		arr2, bdict2 = net.array

		assert np.sum((arr1 - arr2)**2) < epsilon
		assert np.sum((arr1 - arr2)**2) < epsilon

def test_mergeLinks_TreeTensor_Compress():
	for i in range(5):
		net = Network()

		x = np.random.randn(2,2,2,2,3,3)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n1 = Node(xt)
		x = np.random.randn(2,2,2,2,3,3)
		xt = TreeTensor(accuracy=epsilon)
		xt.addTensor(ArrayTensor(x))
		n2 = Node(xt)

		net.addNode(n1)
		Link(n1.buckets[0], n2.buckets[0])
		Link(n1.buckets[1], n2.buckets[1])
		net.addNode(n2)

		arr1, bdict1 = net.array

		net.mergeLinks(n1, compress=True, accuracy=epsilon)

		arr2, bdict2 = net.array

		assert np.sum((arr1 - arr2)**2) < epsilon
		assert np.sum((arr1 - arr2)**2) < epsilon
