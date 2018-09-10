from copy import deepcopy
import numpy as np
import operator
import sys

from TNR.Network.network import Network
from TNR.Network.node import Node
from TNR.Network.bucket import Bucket
from TNR.Network.link import Link
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Utilities.svd import entropy, splitArray

sys.setrecursionlimit(10000)

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeNetwork'])


class TreeNetwork(Network):
    '''
    A treeNetwork is a special case of a Network in which the Network being represented
    contains no cycles. This allows matrix elements of a treeNetwork to be efficiently
    evaluated.

    As the only quantities which matter are the matrix elements, the treeNetwork may
    refactor itself through singular value decomposition (SVD) to minimize memory use, and
    so no assumptions should be made about the Nodes in this object, just the external
    Buckets.

    Internally all Nodes of a treeNetwork have Tensors of rank at most 3.
    SVD factoring is used to enforce this.
    '''

    def __init__(self, accuracy=1e-4):
        '''
        treeNetworks require an accuracy argument which determines how accurately (in terms of relative error)
        they promise to represent their matrix elements.
        '''
        super().__init__()

        self.accuracy = accuracy
    
    def __deepcopy__(self, memodict={}):
        copy = super().__deepcopy__(memodict)
        copy.accuracy = self.accuracy
        return copy
    
    def pathBetween(self, node1, node2, calledFrom=None):
        '''
        Returns the unique path between node1 and node2.
        This is done by treating node1 as the root of the binary tree and performing a depth-first search.
        Note that this search only iterates through the internal buckets in the network: it will not consider
        nodes in another network.
        '''
        if node1 == node2:  # Found it!
            return [node1]

        if len(self.internalConnected(node1)
               ) == 1 and calledFrom is not None:  # Nothing left to search
            return []

        for c in self.internalConnected(node1):  # Search children
            l = node1.findLink(c)
            if c is not calledFrom:
                path = self.pathBetween(c, node2, calledFrom=node1)
                if len(path) > 0:  # Means the recursive call found it
                    path2 = [node1] + path

                    if len(path2) > 0:
                        assert node1 in path2
                        assert node2 in path2

                    return path2

        return []

    def splitNode(self, node, ignore=None):
        '''
        Takes as input a Node and ensures that it is at most rank 3 by factoring rank 3 tensors
        out of it until what remains is rank 3. The factoring is done via a greedy algorithm,
        where the pair of indices with the least correlation with the rest are factored out.
        This is determined by explicitly tracing out all but those indices from the density
        matrix and computing the entropy.

        ignore may be None or a pair of indices.
        In the latter case, the pair of indices will be required to stay together.
        This is enforced by having the pair be the first one factored.
        '''
        nodes = []

        while node.tensor.rank > 3:
            self.removeNode(node)

            array = node.tensor.scaledArray

            s = []

            if ignore is not None:
                p = ignore
                ignore = None
            else:
                p = entropy(array)

            u, v, indices1, indices2 = splitArray(
                array, p, accuracy=self.accuracy)

            if u.shape[-1] > 1:
                b1 = Bucket()
                b2 = Bucket()
                n1 = Node(
                    ArrayTensor(
                        u,
                        logScalar=node.tensor.logScalar /
                        2),
                    Buckets=[
                        node.buckets[i] for i in indices1] +
                    [b1])
                n2 = Node(ArrayTensor(v,
                                      logScalar=node.tensor.logScalar / 2),
                          Buckets=[b2] + [node.buckets[i] for i in indices2])
                # This line has to happen before addNode to prevent b1 and b2
                # from becoming externalBuckets
                _ = Link(b1, b2)
            else:
                # Cut link
                u = u[..., 0]
                v = v[0]
                n1 = Node(
                    ArrayTensor(
                        u,
                        logScalar=node.tensor.logScalar /
                        2),
                    Buckets=[
                        node.buckets[i] for i in indices1])
                n2 = Node(
                    ArrayTensor(
                        v,
                        logScalar=node.tensor.logScalar /
                        2),
                    Buckets=[
                        node.buckets[i] for i in indices2])

            self.addNode(n1)
            self.addNode(n2)
            nodes.append(n1)

            node = n2

        nodes.append(node)

        return nodes

