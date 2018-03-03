from operator import mul
from copy import deepcopy
from collections import defaultdict

import itertools as it
import numpy as np
import operator
import networkx

from TNR.Tensor.tensor import Tensor
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Network.treeNetwork import TreeNetwork
from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.bucket import Bucket
from TNR.Utilities.svd import entropy
from TNR.TensorLoopOptimization.loopOpt import optimizeNorm as optimize
from TNR.TensorLoopOptimization.optimizer import cut


counter0 = 0

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])


class TreeTensor(Tensor):

    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.network = TreeNetwork(accuracy=accuracy)
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
        s = s + 'Tree Tensor with Shape:' + str(self.shape) + ' and Network:\n'
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

    def optimizeLoop(self, loop):
        logger.debug('Optimizing loop.')

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
        arrs = optimize(tensors, self.accuracy)


        # Now transpose back to the original shape
        for i,arr in enumerate(arrs):
            ind0, ind1, ind2 = inds[i]

            ind0 = inds[i].index(0)
            ind1 = inds[i].index(1)
            ind2 = inds[i].index(2)

            arr = np.transpose(arr, axes=(ind0, ind1, ind2))
            assert arr.shape[inds[i][1]] == shs[i]

            loop[i].tensor = ArrayTensor(arr)

        for i,l in enumerate(loop):
            arrM1 = loop[i-1].tensor.array
            arr = loop[i].tensor.array
            arrP1 = loop[(i+1)%len(loop)].tensor.array

            assert arr.shape[inds[i][1]] == shs[i]
            assert arrM1.shape[inds[i-1][2]] == arr.shape[inds[i][0]]
            assert arr.shape[inds[i][2]] == arrP1.shape[inds[(i+1)%len(loop)][0]]

        logger.debug('Optimized.')


    def optimizeLoops(self):
        cycles = networkx.cycles.cycle_basis(self.network.toGraph())

        for c in cycles:
            self.optimizeLoop(c)
            self.network.cutLinks()

    def cutLoop(self, loop):
        logger.debug('Cutting loop.')
        self.network.check()

        print('----')
        print(self)
        print('--')

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

        for i,l in enumerate(loop):
            arrM1 = loop[i-1].tensor.array
            arr = loop[i].tensor.array
            arrP1 = loop[(i+1)%len(loop)].tensor.array

            assert arr.shape[inds[i][1]] == shs[i]
            assert arrM1.shape[inds[i-1][2]] == arr.shape[inds[i][0]]
            assert arr.shape[inds[i][2]] == arrP1.shape[inds[(i+1)%len(loop)][0]]

        print(self)
        print('----')

        self.network.check()
        self.network.cutLinks()
        self.network.check()

        logger.debug('Cut.')


    def eliminateLoops(self):
        while len(networkx.cycles.cycle_basis(self.network.toGraph())) > 0:
            todo = 1
            while todo > 0:
                # Contract rank 2 objects
                self.contractRank2()

                # Contract along double links
                done = set()
                while len(
                    done.intersection(
                        self.network.nodes)) < len(
                        self.network.nodes):
                    n = next(iter(self.network.nodes.difference(done)))
                    nodes = self.network.internalConnected(n)
                    merged = False
                    for n2 in nodes:
                        if len(n.findLinks(n2)) > 1:
                            self.network.mergeNodes(n, n2)
                            merged = True
                    if not merged:
                        done.add(n)
                    print(len(done.intersection(self.network.nodes)), len(self.network.nodes), len(done))

                # See if done
                todo = 0
                for n in self.network.nodes:
                    if n.tensor.rank == 2 and len(self.network.internalConnected(n)):
                        todo += 1
                    for m in self.network.internalConnected(n):
                        if len(n.linksConnecting(m)) > 1:
                            todo += 1
                print(todo)

            cycles = networkx.cycles.cycle_basis(self.network.toGraph())
            if len(cycles) > 0:
                print(len(cycles))
                c = cycles.pop()
                self.cutLoop(c)
                self.contractRank2()

        assert len(networkx.cycles.cycle_basis(self.network.toGraph())) == 0

    def traceOut(self, ind):
        '''
        Traces out the component of the tensor associated with the bucket b.
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
        arr = self.array

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
                # We may be introducing a loop
                _ = Link(b1, b2)
                t.network.externalBuckets.remove(b1)
                t.network.externalBuckets.remove(b2)
                t.network.internalBuckets.add(b1)
                t.network.internalBuckets.add(b2)
                t.eliminateLoops()

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
        This method merges the listed external indices using a tree tensor
        by attaching the identity tensor to all of them and to a new
        external bucket. It then returns the new tree tensor.
        '''

        buckets = [self.externalBuckets[i] for i in inds]
        shape = [self.shape[i] for i in inds]

        # Create identity array
        shape.append(np.product(shape))
        iden = np.identity(shape[-1])
        iden = np.reshape(iden, shape)

        # Create Tree Tensor holding the identity
        tens = ArrayTensor(iden)
        tn = TreeTensor(self.accuracy)
        tn.addTensor(tens)

        # Contract the identity
        ttens = self.contract(inds, tn, list(range(len(buckets))))

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

    def contractRank2(self):
        done = set()
        while len(
            done.intersection(
                self.network.nodes)) < len(
                self.network.nodes):
            n = next(iter(self.network.nodes.difference(done)))
            if n.tensor.rank == 2:
                nodes = self.network.internalConnected(n)
                if len(nodes) > 0:
                    self.network.mergeNodes(n, nodes.pop())
                else:
                    done.add(n)
            else:
                done.add(n)      

    def optimize(self):
        '''
        Optimizes the tensor network to minimize memory usage.
        '''

        logger.info('Optimizing tensor with shape ' + str(self.shape) + '.')

        s2 = 0
        for n in self.network.nodes:
            s2 += n.tensor.size

        logger.info('Stage 1: Contracting Rank-2 Tensors.')
        self.contractRank2()



        logger.info('Stage 2: Contracting Double Links.')
        done = set()
        while len(
            done.intersection(
                self.network.nodes)) < len(
                self.network.nodes):
            n = next(iter(self.network.nodes.difference(done)))
            nodes = self.network.internalConnected(n)
            merged = False
            for n2 in nodes:
                if len(n.findLinks(n2)) > 1:
                    self.network.mergeNodes(n, n2)
                    merged = True
            if not merged:
                done.add(n) 
        logger.info('Stage 3: Optimizing Links.')

        while len(
            self.optimized.intersection(
                self.network.internalBuckets)) < len(
                self.network.internalBuckets):
            b1 = next(
                iter(
                    self.network.internalBuckets.difference(
                        self.optimized)))
            b2 = b1.otherBucket
            n1 = b1.node
            n2 = b2.node

            if n1.tensor.rank < 3 or n2.tensor.rank < 3:
                self.optimized.add(b1)
                self.optimized.add(b2)
                continue

            sh1 = n1.tensor.shape
            sh2 = n2.tensor.shape
            s = n1.tensor.size + n2.tensor.size

            logger.debug('Optimizing tensors ' +
                         str(n1.id) +
                         ',' +
                         str(n2.id) +
                         ' with shapes' +
                         str(n1.tensor.shape) +
                         ',' +
                         str(n2.tensor.shape))

            t, buckets = self.network.dummyMergeNodes(n1, n2)
            arr = t.array
            if n1.tensor.rank == 3:
                ss = set([0, 1])
            elif n2.tensor.rank == 3:
                ss = set([2, 3])
            else:
                ss = None

            logger.debug('Computing minimum entropy cut...')
            best = entropy(arr, pref=ss)
            logger.debug('Done.')

            if set(best) != ss and set(best) != set(
                    range(n1.tensor.rank + n2.tensor.rank - 2)).difference(ss):
                n = self.network.mergeNodes(n1, n2)
                nodes = self.network.splitNode(n, ignore=best)

                for b in nodes[0].buckets:
                    self.optimized.discard(b)
                for b in nodes[1].buckets:
                    self.optimized.discard(b)

                if nodes[0] in nodes[1].connectedNodes:
                    assert len(nodes) == 2
                    l = nodes[0].findLink(nodes[1])
                    self.optimized.add(l.bucket1)
                    self.optimized.add(l.bucket2)
                    logger.debug('Optimizer improved to shapes to' +
                                 str(nodes[0].tensor.shape) +
                                 ',' +
                                 str(nodes[1].tensor.shape))
                else:
                    # This means the link was cut
                    logger.debug('Optimizer cut a link. The resulting shapes are ' +
                                 str(nodes[0].tensor.shape) + ', ' + str(nodes[1].tensor.shape))

            else:
                self.optimized.add(b1)
                self.optimized.add(b2)

            logger.debug('Optimization steps left:' + str(-len(self.optimized.intersection(
                self.network.internalBuckets)) + len(self.network.internalBuckets)))

        s1 = 0
        for n in self.network.nodes:
            s1 += n.tensor.size
        logger.info('Optimized network with shape ' +
                    str(self.shape) +
                    ' and ' +
                    str(len(self.network.nodes)) +
                    ' nodes. Size reduced from ' +
                    str(s2) +
                    ' to ' +
                    str(s1) +
                    '.')
