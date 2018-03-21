import numpy as np

from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TreeTensor.treeTensor import TreeTensor

epsilon = 1e-10


def test_init():
    x = np.random.randn(2, 3, 3)
    xt = ArrayTensor(x)
    n = Node(xt)

    assert len(n.buckets) == xt.rank
    for i, b in enumerate(n.buckets):
        assert b.node == n
        assert b.index == i
        assert n.bucketIndex(b) == i


def test_links():
    x = np.random.randn(2, 3, 3)
    xt = ArrayTensor(x)
    n1 = Node(xt)

    x = np.random.randn(2, 3, 3)
    xt = ArrayTensor(x)
    n2 = Node(xt)

    l1 = Link(n1.buckets[0], n2.buckets[0])

    assert n1.linkedBuckets[0].otherBucket == n2.buckets[0]

    l2 = Link(n2.buckets[1], n1.buckets[1])

    assert n1.linkedBuckets[1].otherBucket == n2.buckets[1]

    assert l1.bucket1.node == n1 or l1.bucket2.node == n1
    assert l1.bucket1.node == n2 or l1.bucket2.node == n2

    assert l1 in n1.findLinks(n2)
    assert l2 in n1.findLinks(n2)
    assert l1 in n2.findLinks(n1)
    assert l2 in n2.findLinks(n1)

    assert n1.findLink(n2) == l1 or n1.findLink(n2) == l2

    assert n1.indexConnecting(n2) == 0 or n1.indexConnecting(n2) == 1
    assert n2.indexConnecting(n1) == 0 or n2.indexConnecting(n1) == 1

    assert 0 in n1.indicesConnecting(n2)[0]
    assert 1 in n1.indicesConnecting(n2)[0]
    assert 2 not in n1.indicesConnecting(n2)[0]

    assert 0 in n2.indicesConnecting(n1)[0]
    assert 1 in n2.indicesConnecting(n1)[0]
    assert 2 not in n2.indicesConnecting(n1)[0]

    assert 0 in n1.indicesConnecting(n2)[1]
    assert 1 in n1.indicesConnecting(n2)[1]
    assert 2 not in n1.indicesConnecting(n2)[1]

    assert 0 in n2.indicesConnecting(n1)[1]
    assert 1 in n2.indicesConnecting(n1)[1]
    assert 2 not in n2.indicesConnecting(n1)[1]

    assert len(n1.connectedNodes) == 1
    assert n2 in n1.connectedNodes


def test_mergeBuckets():
    x = np.random.randn(2, 3, 3)
    xt = ArrayTensor(x)
    n = Node(xt)

    buckets = list(n.buckets)

    b = n.mergeBuckets(n.buckets[1:])

    assert n.tensor.rank == 2
    assert n.tensor.shape == (2, 9)
    assert len(n.buckets) == 2
    assert n.buckets[0] == buckets[0]
    assert n.buckets[1] != buckets[1]
    assert n.buckets[1] == b
    assert np.sum((np.reshape(x, (-1, 9)) - n.tensor.array)**2) < epsilon

    x = np.random.randn(2, 3, 3)
    xt = TreeTensor(accuracy=epsilon)
    xt.addTensor(ArrayTensor(x))
    n = Node(xt)

    buckets = list(n.buckets)

    b = n.mergeBuckets(n.buckets[1:])

    assert n.tensor.rank == 2
    assert n.tensor.shape == (2, 9)
    assert len(n.buckets) == 2
    assert n.buckets[0] == buckets[0]
    assert n.buckets[1] != buckets[1]
    assert n.buckets[1] == b
    assert np.sum((np.reshape(x, (-1, 9)) - n.tensor.array)**2) < epsilon

    x = np.random.randn(2, 2, 2, 2, 2, 2)
    xt = TreeTensor(accuracy=epsilon)
    xt.addTensor(ArrayTensor(x))
    n = Node(xt)

    buckets = list(n.buckets)

    b = n.mergeBuckets(n.buckets[4:])

    assert n.tensor.rank == 5
    assert n.tensor.shape == (2, 2, 2, 2, 4)
    assert len(n.buckets) == 5
    assert n.buckets[0] == buckets[0]
    assert n.buckets[-1] != buckets[-1]
    assert n.buckets[-1] == b
    assert np.sum((np.reshape(x, (2, 2, 2, 2, 4)) -
                   n.tensor.array)**2) < epsilon
