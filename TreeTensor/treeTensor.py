from operator import mul
from copy import deepcopy
from collections import defaultdict

import itertools as it
import numpy as np
import operator
import networkx

from TNR.Tensor.tensor import Tensor
from TNR.NetworkTensor.networkTensor import NetworkTensor
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Network.treeNetwork import TreeNetwork
from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.bucket import Bucket
from TNR.Utilities.svd import entropy
from TNR.TensorLoopOptimization.optimizer import cut


counter0 = 0

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])


class TreeTensor(NetworkTensor):

    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.network = TreeNetwork(accuracy=accuracy)
        self.externalBuckets = []
        self.optimized = set()

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

    def contract(self, ind, other, otherInd, front=True):
        # We copy the two networks first. If the other tensor is an ArrayTensor
        # we cast it to a TreeTensor first. If the other tensor is a NetworkTensor
        # there's no problem: we're going to eliminate all loops at the end anyway.

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

                ### There's a bug here somewhere that isn't in the cutter so... gotta fix that.

                # See if done
                todo = 0
                for n in self.network.nodes:
                    if n.tensor.rank == 2 and len(self.network.internalConnected(n)):
                        todo += 1
                    for m in self.network.internalConnected(n):
                        if len(n.linksConnecting(m)) > 1:
                            todo += 1

            cycles = networkx.cycles.cycle_basis(self.network.toGraph())
            if len(cycles) > 0:
                c = cycles.pop()
                self.cutLoop(c)
                self.contractRank2()

        assert len(networkx.cycles.cycle_basis(self.network.toGraph())) == 0

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

        t = super().trace(ind0, ind1)
        t.eliminateLoops()
        return t

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
