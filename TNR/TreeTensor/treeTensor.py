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
from TNR.TensorLoopOptimization.optimizer import optimize as opt
from TNR.TensorLoopOptimization.svdCut import svdCut
from TNR.TensorLoopOptimization.densityMatrix import cutSVD
from TNR.Utilities.graphPlotter import plot, plotGraph
from TNR.Utilities.linalg import L2error
from TNR.Utilities.misc import shortest_cycles
from TNR.Environment.environment import artificialCut, identityEnvironment, fullEnvironment

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

    def __deepcopy__(self, memodict={}):
        copy = super().__deepcopy__(memodict)
        copy.optimized = set() # TODO: Fix this implementation
        return copy

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
        print(len(loop))
        self.network.check()

        # Form the environment network

#        environment, net, internalBids, envBids = artificialCut(self, loop)
        environment, net, internalBids, envBids = identityEnvironment(self, loop)
        bids = list([b.id for b in net.externalBuckets])

        # Determine optimal cut
        ranks, costs, lids = cutSVD(net, environment, self.accuracy, bids, envBids)

        # Assocaite links with ranks
        links = set()
        for n in net.network.nodes:
            for b in n.buckets:
                if b.linked:
                    links.add(b.link)
    
        freeBucket = lambda x: list(b for b in x.buckets if not b.linked)[0]
        rankDict = {}
        for i,l in enumerate(links):
            for j,l2 in enumerate(links):
                if i != j:
                    n11 = l.bucket1.node
                    n12 = l.bucket2.node
                    n21 = l2.bucket1.node
                    n22 = l2.bucket2.node


                    leftBids = set()
                    current = n11
                    prev = n12
                    while True:
                        leftBids.add(freeBucket(current).id)
                        if current == n21 or current == n22:
                            break
                        found = False
                        for n in current.connectedNodes:
                            if n != prev:
                                prev = current
                                current = n
                                found = True
                                break

                        if not found:
                            break

                    rightBids = set(b.id for b in net.externalBuckets).difference(leftBids)

                    leftBids = frozenset(leftBids)
                    rightBids = frozenset(rightBids)

                    rankDict[(leftBids, rightBids)] = ranks[lids.index(l.id), lids.index(l2.id)]
                    rankDict[(rightBids, leftBids)] = ranks[lids.index(l.id), lids.index(l2.id)]


        ind = np.argmin(costs)
        
        ranks = ranks[ind]
        ranks[ranks == 0] = 1
        
        logger.debug('Final ranks: ' + str(ranks))

        # Cut
        for i,r in enumerate(ranks):
            if r == 1:
                lid = lids[i]
        for n in net.network.nodes:
            for b in n.buckets:
                if b.linked and b.link.id == lid:
                    l = b.link

        #print('Err',net.array)
        newNet = svdCut(net, environment, l, bids, envBids, rankDict)
        #print('Err_new',newNet.array)



        ranks2 = []
        doneLinks = set()
        for n in newNet.network.nodes:
            for b in n.buckets:
                if b.linked and b.link not in doneLinks:
                    doneLinks.add(b.link)
                    ranks2.append(b.size)

        logger.debug('actual ranks: ' + str(ranks2) + ', predicted: ' + str(ranks))

        # Now put the new nodes in the network and replace the loop
        #print('Err_0',self.array)
        #print(self)
        netBids = list(b.id for b in net.externalBuckets)

        toRemove = []
        removedBuckets = []
        for n in self.network.nodes:
            nbids = set(b.id for b in n.buckets)
            if len(nbids.intersection(netBids)) > 0:
                toRemove.append(n)
        for n in toRemove:
            self.network.removeNode(n)
            removedBuckets.extend(n.buckets)
                        
        existingBuckets = {}
        for b in removedBuckets:
            existingBuckets[b.id] = b
        
        newNodes = list(newNet.network.nodes)
        for n in newNodes:
            newNet.network.removeNode(n)
            
            newBuckets = []
            for b in n.buckets:
                if b.id in netBids:
                    oldB = existingBuckets[b.id]
                    newBuckets.append(oldB)
                else:
                    newBuckets.append(b)

            n.buckets = newBuckets
            for b in n.buckets:
                b.node = n

            self.network.addNode(n)
        
        #print(self)
        #print('Err_1',self.array)
        
        self.network.cutLinks()
        self.network.check()

        logger.debug('Cut.')


    def eliminateLoops(self):
        canon = lambda x: list(y for y in self.network.nodes for i in range(len(x)) if y.id == x[i])
        prodkey = lambda x: sum(x[i].tensor.size*x[i+1].tensor.size for i in range(len(x)-1))
        while len(networkx.cycles.cycle_basis(self.network.toGraph())) > 0:
#        while len(shortest_cycles(self.network.toGraph())) > 0:

            self.contractRank2()

            cycles = sorted(networkx.cycles.cycle_basis(self.network.toGraph()), key=len)
#            cycles = sorted(list(map(canon,shortest_cycles(self.network.toGraph()))), key=prodkey)
            if len(cycles) > 0:
                print('Cycles:',len(cycles), list(len(c) for c in cycles))
                old_nodes = set(self.network.nodes)

                #arr = self.array

                self.cutLoop(cycles[0])

                #arr2 = self.array
                #print('Original (Err 2):',arr,arr2)

                self.contractRank2()
                new_nodes = set(self.network.nodes)

                affected = set(cycles[0])
                affected.update(new_nodes.difference(old_nodes))
                self.network.graph = None
#                cycles = sorted(list(map(canon,shortest_cycles(self.network.toGraph()))), key=prodkey)
                cycles = sorted(networkx.cycles.cycle_basis(self.network.toGraph()), key=len)

                # Really want to go over all small cycles, but unclear how to generate them.
                #cycles = networkx.cycles.simple_cycles(networkx.DiGraph(self.network.toGraph()))

                for loop in cycles:
                    if len(affected.intersection(loop)) > 0 and len(loop) < 0:

                        print(loop)

                        #print('Optimizing cycle of length',len(loop))

                        environment, net, internalBids, envBids = identityEnvironment(self, loop)

                        print(net)

                        #environment, net, internalBids, envBids = artificialCut(self, loop)
                        bids = list([b.id for b in net.externalBuckets])

                        #print(net)

                        # Optimize
                        #arr = self.array #####
                        #arr0 = net.array
                        #net0 = deepcopy(net)
                        net, inds = opt(net, 1e-12, environment, bids, envBids)
                        #bidDict = {b.id:i for i,b in enumerate(net.externalBuckets)}
                        #iDict = {i:bid for i,bid in enumerate(bids)}
                        #net.externalBuckets = list(net.externalBuckets[bidDict[iDict[i]]] for i in range(len(bids)))
                        #arr01 = net.array

                        #print('Err_internal',L2error(arr0,arr01))

                        # Throw the new tensors back in
                        num = 0
                        for m in self.network.nodes:
                            for n in net.network.nodes:
                                if n.id == m.id:
                                    m.tensor = n.tensor
                                    num += 1

                        #environment, net, internalBids, envBids = identityEnvironment(self, loop)
                        #bidDict = {b.id:i for i,b in enumerate(net.externalBuckets)}
                        #iDict = {i:bid for i,bid in enumerate(bids)}
                        #net.externalBuckets = list(net.externalBuckets[bidDict[iDict[i]]] for i in range(len(bids)))                        
                        #arr11 = net.array
                        #print(arr0 / np.max(np.abs(arr0)))
                        #print(arr11 / np.max(np.abs(arr0)))

                        #print('Err_internal_set',L2error(arr0,arr11))
                        # Either something is going wrong with the above insertion procedure
                        # or else the network is somehow hypersensitive to small components.
                        # That couuld be captured by the environment but the degree (of order
                        # 1e15) is surprising, especially on small networks where Z ~ 5000.
                        # Is something going wrong with the above insertion proceduure?

                        #arr2 = self.array  #####
                        #print('Original (Err):',arr,arr2)
                        #environment, net2, internalBids, envBids = fullEnvironment(self, loop)
                        #print('Err Angle:',np.exp(self.logNorm - environment.logNorm - net2.logNorm))
                        #print('Err Angle:',np.exp(self.logNorm - environment.logNorm - net0.logNorm))
                        #print('Err2',L2error(arr,arr2))

                        assert num == len(loop)



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
