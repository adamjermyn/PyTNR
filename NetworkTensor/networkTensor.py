from operator import mul
from copy import deepcopy
from collections import defaultdict

import itertools as it
import numpy as np
import operator
import networkx

from TNR.Tensor.tensor import Tensor
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Network.network import network
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
        self.network = network(accuracy=accuracy)
        self.externalBuckets = []
        self.optimized = set()

    def addTensor(self, tensor):
        n = Node(tensor, Buckets=[Bucket() for _ in range(tensor.rank)])
        self.network.addNode(n)
        self.externalBuckets.extend(n.buckets)
        if tensor.rank > 3:
            self.network.splitNode(n)
        return n

    def __str__(self):
        s = ''
        s = s + 'Network Tensor with Shape:' + str(self.shape) + ' and Network:\n'
        s = s + str(self.network)
        return s

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

    def contract(self, ind, other, otherInd, front=True):
        # This method could be vastly simplified by defining a cycle basis
        # class

        # We copy the two networks first. If the other is an ArrayTensor we
        # cast it to a TreeTensor first.
        t1 = deepcopy(self)
        if hasattr(other, 'network'):
            t2 = deepcopy(other)
        else:
            t2 = TreeTensor(self.accuracy)
            t2.addTensor(other)

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

        # Merge any rank-1 or rank-2 objects
        done = set()
        while len(done.intersection(t1.network.nodes)) < len(t1.network.nodes):
            n = next(iter(t1.network.nodes.difference(done)))
            if n.tensor.rank <= 2:
                nodes = t1.network.internalConnected(n)
                if len(nodes) > 0:
                    t1.network.mergeNodes(n, nodes.pop())
                else:
                    done.add(n)
            else:
                done.add(n)

        t1.externalBuckets = extB
        assert t1.network.externalBuckets == set(t1.externalBuckets)

        for n in t1.network.nodes:
            assert n.tensor.rank <= 3

        assert t1.rank == self.rank + other.rank - 2 * len(ind)

        return t1

    def cutLoop(self, loop):
        logger.debug('Cutting loop.')
        self.network.check()

        prev = self.array
        prevL = list([l.tensor.shape for l in loop])


        loopNetwork = self.network.copySubset(looo)
        

        # Get tensors and transpose into correct form
        tensors = []
        inds = []
        shs = []
        for i,l in enumerate(loop):
            arr = l.tensor.array
            ind0 = l.indexConnecting(loop[i-1])
            ind2 = l.indexConnecting(loop[(i+1)%len(loop)])
            ind1 = set((0,1,2)).difference(set((ind0,ind2))).pop()
            shs.append(arr.shape[ind1])
            inds.append((ind0, ind1, ind2))
            arr = np.transpose(arr, axes=(ind0, ind1, ind2))
            tensors.append(arr)

        # Optimize
        arrs = cut(tensors, self.accuracy)

        # Now transpose back to the original shape
        for i,arr in enumerate(arrs):
            ind0, ind1, ind2 = inds[i]

            ind0 = inds[i].index(0)
            ind1 = inds[i].index(1)
            ind2 = inds[i].index(2)

            arr = np.transpose(arr, axes=(ind0, ind1, ind2))
            assert arr.shape[inds[i][1]] == shs[i]

            loop[i].tensor = ArrayTensor(arr)

        new = self.array

        err = np.sum((prev - new)**2) / np.sum(prev**2)

        if err > 3 * self.accuracy:

            ### There's a bug in loop optimizing which occasionally produces
            # considerably larger-than-expected accuracy.

            print(prev)
            print(new)
            print(err)
            print(err / self.accuracy)
            print(prevL)
            print(list([l.tensor.shape for l in loop]))
            exit()

        for i,l in enumerate(loop):
            arrM1 = loop[i-1].tensor.array
            arr = loop[i].tensor.array
            arrP1 = loop[(i+1)%len(loop)].tensor.array

            assert arr.shape[inds[i][1]] == shs[i]
            assert arrM1.shape[inds[i-1][2]] == arr.shape[inds[i][0]]
            assert arr.shape[inds[i][2]] == arrP1.shape[inds[(i+1)%len(loop)][0]]



        self.network.cutLinks()

        self.network.check()

        logger.debug('Cut.')

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

