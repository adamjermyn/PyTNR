from operator import mul
from copy import deepcopy
from collections import defaultdict

import itertools as it
import numpy as np
import operator
import networkx
from random import shuffle

from TNR.NetworkTensor.networkTensor import NetworkTensor
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Network.treeNetwork import TreeNetwork
from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.bucket import Bucket
from TNR.Utilities.svd import entropy
from TNR.TensorLoopOptimization.optimizer import cut
from TNR.TensorLoopOptimization.densityMatrix import cutSVD

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
        s = 'Tree Tensor with Shape:' + str(self.shape) + ' and Network:\n'
        s = s + str(self.network)
        return s

    def promote(self, other):
        if not hasattr(other, 'network'):
            t = TreeTensor(self.accuracy)
            t.addTensor(other)
        else:
            t = deepcopy(other)
        return t

    def artificialCut(self, exclude):
        '''
        Copies the network, cuts all but one loop, and removes the ndoes in that loop.
        '''
        
        t = deepcopy(self)
        g = t.network.toGraph()
        basis = networkx.cycles.cycle_basis(t.network.toGraph())

        correction = 1.
        excludeIDs = list(n.id for n in exclude)

        assert set(t.externalBuckets) == set(t.network.externalBuckets)

        # Sometimes it is necessary to cut a bond running from the loop
        # to an adjacent tensor (this is needed only when a single tensor
        # outside the loop joins two tensors in the loop). In this case
        # the later logic will attempt to recreate this bond, so we have
        # to explicitly track the bucket ID's involved.
        onLoopBids = set()

        while len(basis) > 1:
            shuffle(basis)
            cycle = basis[0]
            done = False
            
            for i in range(len(cycle)):
                if cycle[i-1].id not in excludeIDs and cycle[i].id not in excludeIDs:
                    link = cycle[i-1].findLink(cycle[i])
                    correction *= link.bucket1.size
                    t.network.removeLink(link)
                    t.externalBuckets.append(link.bucket1)
                    t.externalBuckets.append(link.bucket2)
                    done = True
                    break
                    
            if not done:
                for i in range(len(cycle)):
                    if cycle[i-1].id not in excludeIDs or cycle[i].id not in excludeIDs:
                        link = cycle[i-1].findLink(cycle[i])
                        onLoopBids.add(link.bucket1.id)
                        onLoopBids.add(link.bucket2.id)
                        correction *= link.bucket1.size
                        t.network.removeLink(link)
                        t.externalBuckets.append(link.bucket1)
                        t.externalBuckets.append(link.bucket2)
                        done = True
                        break

            
            assert set(t.externalBuckets) == set(t.network.externalBuckets)

            g = t.network.toGraph()
            basis = networkx.cycles.cycle_basis(t.network.toGraph())

        nodes2 = list(t.network.nodes)

        for n in nodes2:
            if n.id in excludeIDs:
                t.removeNode(n)

        for b in t.externalBuckets:
            b.link = None

        for nid in excludeIDs:
            assert nid not in set(n.id for n in t.network.nodes)

        return t, correction, onLoopBids


    def contract(self, ind, other, otherInd, front=True, elimLoops=True):
        t = super().contract(ind, other, otherInd, front=front)
        if elimLoops:
            # Merge any rank-1 or rank-2 objects
            done = set()
            while len(done.intersection(t.network.nodes)) < len(t.network.nodes):
                n = next(iter(t.network.nodes.difference(done)))
                if n.tensor.rank <= 2:
                    nodes = t.network.internalConnected(n)
                    if len(nodes) > 0:
                        t.network.mergeNodes(n, nodes.pop())
                    else:
                        done.add(n)
                else:
                    done.add(n)
            t.eliminateLoops()
        return t

    def cutLoop(self, loop, cutIndex=None):
        logger.debug('Cutting loop.')
        self.network.check()

        # Form the loop network
        net = self.copySubset(loop)
        bids = list([b.id for b in net.externalBuckets])

        # Form the environment network
        environment, correction, onLoopBids = self.artificialCut(loop)
        assert environment.network.externalBuckets == set(environment.externalBuckets)

        # Associate bucket indices between the loop and the environment
        otherBids = []
        for i,b in enumerate(net.externalBuckets):
            ind = list(l.id for l in loop).index(b.node.id)
            ind2 = list(b2.id for b2 in loop[ind].buckets).index(b.id)
            if loop[ind].buckets[ind2].linked and b.id not in onLoopBids:
                otherBids.append(loop[ind].buckets[ind2].otherBucket.id)
            else:
                n = Node(ArrayTensor(np.identity(net.externalBuckets[i].size)))
                environment.network.addNode(n)
                environment.externalBuckets.append(n.buckets[0])
                environment.externalBuckets.append(n.buckets[1])
                # We've just added two buckets, so we associate one with the loop
                # and one with the environment
                otherBids.append(n.buckets[0].id)
                
        assert environment.network.externalBuckets == set(environment.externalBuckets)
        
        # Testing code
        ranks, costs, lids = cutSVD(net, environment, self.accuracy, bids, otherBids)
        
        ind = np.argmin(costs)
        
        ranks = ranks[ind]
        ranks[ranks == 0] = 1

        # Optimize
        correction = 1.
        logger.debug('Correction factor:' + str(correction))
        net, inds = cut(net, self.accuracy / correction, environment, bids, otherBids, ranks, lids)
        logger.debug('Correction factor:' + str(correction))

        # Throw the new tensors back in
        num = 0
        for n in net.network.nodes:
            for m in self.network.nodes:
                bidsn = list(b.id for b in n.buckets)
                bidsm = list(b.id for b in m.buckets)
                if len(set(bidsn).intersection(bidsm)) > 0:
                    m.tensor = n.tensor
                    num += 1

        assert num == len(loop)

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

            cycles = sorted(list(networkx.cycles.cycle_basis(self.network.toGraph())), key=len)
            if len(cycles) > 0:
                print(len(cycles), list(len(c) for c in cycles))
                self.cutLoop(cycles[0])
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
  

    def optimize(self):
        '''
        Optimizes the tensor network to minimize memory usage.
        '''

        logger.info('Optimizing tensor with shape ' + str(self.shape) + '.')

        s2 = 0
        for n in self.network.nodes:
            s2 += n.tensor.size

        logger.info('Stage 1: Contracting Rank-2 Tensors and Double Links.')
        self.contractRank2()

        logger.info('Stage 2: Optimizing Links.')

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
