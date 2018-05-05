import numpy as np

from TNR.Network.treeNetwork import TreeNetwork
from TNR.Network.network import Node
from TNR.Network.link import Link
from TNR.Tensor.arrayTensor import ArrayTensor

epsilon = 1e-10


def test_pathing():
    tn = TreeNetwork(accuracy=epsilon)

    n1 = Node(ArrayTensor(np.random.randn(3, 3, 3)))

    tn.addNode(n1)

    assert len(tn.pathBetween(n1, n1)) == 1

    n2 = Node(ArrayTensor(np.random.randn(3, 3, 3)))

    Link(n1.buckets[0], n2.buckets[0])

    tn.addNode(n2)

    assert len(tn.pathBetween(n1, n2)) == 2
    assert len(tn.pathBetween(n2, n1)) == 2
    assert tn.pathBetween(n1, n2) == [n1, n2]
    assert tn.pathBetween(n2, n1) == [n2, n1]

    n3 = Node(ArrayTensor(np.random.randn(3, 3, 3)))

    Link(n3.buckets[0], n2.buckets[2])

    tn.addNode(n3)

    assert len(tn.pathBetween(n1, n3)) == 3
    assert len(tn.pathBetween(n3, n1)) == 3
    assert tn.pathBetween(n1, n3) == [n1, n2, n3]
    assert tn.pathBetween(n3, n1) == [n3, n2, n1]
    assert tn.pathBetween(n2, n3) == [n2, n3]
    assert tn.pathBetween(n3, n2) == [n3, n2]
