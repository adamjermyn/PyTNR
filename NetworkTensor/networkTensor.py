from operator import mul
from copy import deepcopy
from collections import defaultdict
from scipy.stats import ortho_group

import itertools as it
import numpy as np
import operator
import networkx

from TNR.Tensor.tensor import Tensor
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Network.network import Network
from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.bucket import Bucket
from TNR.Utilities.svd import entropy

counter0 = 0

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])


class NetworkTensor(Tensor):

    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.network = Network()
        self.externalBuckets = []

    def copy(self):
        new = deepcopy(self)
        for n in new.network.nodes:
            n.id = Node.newid()
        for b in new.network.buckets:
            b.id = Bucket.newid()
        for b in new.externalBuckets:
            b.link = None
        return new

    def addTensor(self, tensor):
        n = Node(tensor, Buckets=[Bucket() for _ in range(tensor.rank)])
        self.network.addNode(n)
        self.externalBuckets.extend(n.buckets)
        if tensor.rank > 3:
            self.network.splitNode(n)
        return n

    def removeNode(self, node):
        assert node in self.network.nodes

        # The external buckets associated with the bonds broken by
        # removing this node are added at the end of the list.
        for b in node.buckets:
            if b not in self.externalBuckets:
                self.externalBuckets.append(b.otherBucket)
                b.otherBucket.link = None
            else:
                self.externalBuckets.remove(b)

        self.network.removeNode(node)

    def __str__(self):
        s = 'Network Tensor with Shape:' + str(self.shape) + ' and Network:\n'
        s = s + str(self.network)
        return s

    def promote(self, other):
        if not hasattr(other, 'network'):
            t = NetworkTensor(self.accuracy)
            t.addTensor(other)
        else:
            t = deepcopy(other)
        return t

    @property
    def array(self):

        arr, logAcc, bdict = self.network.array

        arr *= np.exp(logAcc)

        perm = []
        blist = [b.id for b in self.externalBuckets]

        for b in blist:
            perm.append(bdict[b])

        arr = np.transpose(arr, axes=perm)

        assert arr.shape == tuple(self.shape)
        return arr

    @property
    def shape(self):
        return tuple([b.node.tensor.shape[b.index]
                      for b in self.externalBuckets])

    @property
    def rank(self):
        return len(self.externalBuckets)

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def compressedSize(self):
        size = 0
        for n in self.network.nodes:
            size += n.tensor.size
        return size

    def distBetween(self, ind1, ind2):
        n1 = self.externalBuckets[ind1].node
        n2 = self.externalBuckets[ind2].node
        return len(self.network.pathBetween(n1, n2))

    def distBetweenBuckets(self, b1, b2):
        n1 = b1.node
        n2 = b2.node
        return len(self.network.pathBetween(n1, n2))

    def insertRandomUnitary(self):
        arr0 = self.array

        for n1 in self.network.nodes:
            for n2 in self.network.internalConnected(n1):
                for l in n1.linksConnecting(n2):
                    b1, b2 = l.bucket1, l.bucket2
                    if b1 not in n1.buckets:
                        b1, b2 = b2, b1
                    size = b1.size
                    u = ortho_group.rvs(size)
                    v = np.transpose(u)
                    n1.tensor = n1.tensor.contract(b1.index, ArrayTensor(u), 0)
                    n1.buckets.remove(b1)
                    n1.buckets.append(b1)
                    n2.tensor = n2.tensor.contract(b2.index, ArrayTensor(v), 1)
                    n2.buckets.remove(b2)
                    n2.buckets.append(b2)

        arr1 = self.array

        assert np.sum((arr0 - arr1)**2) / np.sum(arr0**2) < 1e-10


    def contract(self, ind, other, otherInd, front=True):
        # We copy the two networks first. If the other is an ArrayTensor we
        # cast it to a NetworkTensor first.
        t1 = deepcopy(self)
        t2 = self.promote(other)

        # If front == True then we contract t2 into t1, otherwise we contract t1 into t2.
        # This is so we get the index order correct. Thus
        if not front:
            t1, t2 = t2, t1
            otherInd, ind = ind, otherInd

        # Link the networks
        links = []
        for i, j in zip(*(ind, otherInd)):
            b1, b2 = t1.externalBuckets[i], t2.externalBuckets[j]
            assert b1 in t1.network.buckets and b1 not in t2.network.buckets
            assert b2 in t2.network.buckets and b2 not in t1.network.buckets
            links.append(Link(b1, b2))
        # Determine new external buckets list
        for l in links:
            t1.externalBuckets.remove(l.bucket1)
            t2.externalBuckets.remove(l.bucket2)

        extB = t1.externalBuckets + t2.externalBuckets
        # Merge the networks
        toRemove = set(t2.network.nodes)

        for n in toRemove:
            t2.network.removeNode(n)

        for n in toRemove:
            t1.network.addNode(n)

        t1.externalBuckets = extB
        assert t1.network.externalBuckets == set(t1.externalBuckets)

        for n in t1.network.nodes:
            assert n.tensor.rank <= 3

        assert t1.rank == self.rank + other.rank - 2 * len(ind)

        return t1

    def traceOut(self, ind):
        '''
        Traces out the component of the tensor associated with the specified index (ind).
        '''

        t = deepcopy(self)
        b = t.externalBuckets[ind]
        t.network.traceOut(b)
        t.externalBuckets.remove(b)
        return t


    def trace(self, ind0, ind1):
        '''
        Takes as input:
                ind0	-	A list of indices on one side of their Links.
                ind1	-	A list of indices on the other side of their Links.

        Elements of ind0 and ind1 must correspond, such that the same Link is
        represented by indices at the same location in each list.

        Elements of ind0 should not appear in ind1, and vice-versa.

        Returns a Tensor containing the trace over all of the pairs of indices.
        '''

        ind0 = list(ind0)
        ind1 = list(ind1)

        t = deepcopy(self)

        for i in range(len(ind0)):
            b1 = t.externalBuckets[ind0[i]]
            b2 = t.externalBuckets[ind1[i]]

            n1 = b1.node
            n2 = b2.node

            if n1 == n2:
                # So we're just tracing an arrayTensor.
                n1.tensor = n1.tensor.trace([b1.index], [b2.index])
                n1.buckets.remove(b1)
                n1.buckets.remove(b2)
            else:
                # We're connecting two leaves
                _ = Link(b1, b2)
                t.network.externalBuckets.remove(b1)
                t.network.externalBuckets.remove(b2)
                t.network.internalBuckets.add(b1)
                t.network.internalBuckets.add(b2)

            t.externalBuckets.remove(b1)
            t.externalBuckets.remove(b2)

            for j in range(len(ind0)):
                d0 = 0
                d1 = 0

                if ind0[j] > ind0[i]:
                    d0 += 1
                if ind0[j] > ind1[i]:
                    d0 += 1

                if ind1[j] > ind0[i]:
                    d1 += 1
                if ind1[j] > ind1[i]:
                    d1 += 1

                ind0[j] -= d0
                ind1[j] -= d1

        return t

    def flatten(self, inds):
        '''
        This method merges the listed external indices using
        by attaching the identity tensor to all of them and to a new
        external bucket. It then returns the new tree tensor.
        '''

        buckets = [self.externalBuckets[i] for i in inds]
        shape = [self.shape[i] for i in inds]

        # Create identity array
        shape.append(np.product(shape))
        iden = np.identity(shape[-1])
        iden = np.reshape(iden, shape)

        # Create Network Tensor holding the identity
        tens = ArrayTensor(iden)
        nn = NetworkTensor(self.accuracy)
        nn.addTensor(tens)

        # Contract the identity
        ttens = self.contract(inds, nn, list(range(len(buckets))))

        shape2 = [self.shape[i] for i in range(self.rank) if i not in inds]
        shape2.append(shape[-1])
        for i in range(len(shape2)):
            assert ttens.shape[i] == shape2[i]

        return ttens

    def getIndexFactor(self, ind):
        return self.externalBuckets[ind].node.tensor.scaledArray, self.externalBuckets[ind].index

    def setIndexFactor(self, ind, arr):
        tt = deepcopy(self)
        tt.externalBuckets[ind].node.tensor = ArrayTensor(
            arr, logScalar=tt.externalBuckets[ind].node.tensor.logScalar)
        return tt

    def copySubset(self, nodes):
        t = deepcopy(self)

        # Prune the network down to just the specified nodes
        ids = list([n.id for n in nodes])
        nodes2 = list(t.network.nodes)

        for n in nodes2:
            if n.id not in ids:
                t.removeNode(n)

        for b in t.externalBuckets:
            b.link = None

        return t

    def contractRank2(self):
        done = set()
        while len(
            done.intersection(
                self.network.nodes)) < len(
                self.network.nodes):

            n = next(iter(self.network.nodes.difference(done)))

            nodes = self.network.internalConnected(n)
            if len(nodes) > 0:
                n2 = next(iter(nodes))

            if len(nodes) == 0:
                done.add(n)
            elif n.tensor.rank <= 2:
                self.network.mergeNodes(n, nodes.pop())
            elif len(nodes) == 1 and len(n.findLinks(n2)) > 1:
                self.network.mergeNodes(n, n2)
            else:
                done.add(n)

