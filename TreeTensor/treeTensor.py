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
        Copies the network without the excluded nodes and cuts a bond in each loop.
        '''

        t = self.copySubset(set(self.network.nodes).difference(exclude))
        g = t.network.toGraph()
        basis = networkx.cycles.cycle_basis(self.network.toGraph())
        excludeIDs = list(n.id for n in exclude)
        for cycle in basis:
            for i in range(len(cycle)):
                if cycle[i-1].id not in excludeIDs and cycle[i].id not in excludeIDs:
                    link = cycle[i-1].findLinks(cycle[i])
                    t.network.removeLink(link)
                    break
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

    def cutLoop(self, loop):
        logger.debug('Cutting loop.')
        self.network.check()

        prev = self.array
        prevL = list([l.tensor.shape for l in loop])

        # Form the loop network
        net = self.copySubset(loop)
        bids = list([b.id for b in net.externalBuckets])
        otherBids = list([b.otherBucket.id if b.linked else None for b in net.externalBuckets])

        # Form the environment network
        environment = self.artificialCut(loop)
        for i in range(len(bids)):
            if otherBids[i] is None: # If a loop tensor had an external bucket
                n = Node(ArrayTensor(np.identity(net.externalBuckets[i].size)))
                environment.network.addNode(n)
                environment.externalBuckets.append(n.buckets[0])
                environment.externalBuckets.append(n.buckets[1])
                # We've just added two buckets, so we associate one with the loop
                # and one with the environment
                otherBids[i] = n.buckets[0].id

        netArr = net.array
        netA = np.sum(net.array**2)

        # Optimize
        print('HMMM')
        net, inds, err_l2 = cut(net, self.accuracy, environment, bids, otherBids)
        print('HMMM')
        logger.debug('HGMMMM')

        netArr2 = net.array
        netA2 = np.sum(net.array**2)
        bids2 = list([b.id for b in net.externalBuckets])

        trans = list(bids.index(b) for b in bids2)
        netArr = np.transpose(netArr, axes=trans)

        err_frob = np.sum((netArr - netArr2)**2)/np.sum(netArr**2)

        if err_frob > 3 * abs(err_l2):
            ### Ok so the loop cutter is not putting things back the way they began.

            print('__________')
            print(netArr)
            print(netArr2)
            print(bids)
            print(bids2)
            print(netA, netA2, np.dot(np.reshape(netArr,(-1,)), np.reshape(netArr2,(-1,))))
            print(err_frob)
            print(err_l2)
            import matplotlib.pyplot as plt
            plt.plot(np.reshape(netArr,(-1,)),label='Original')
            plt.plot(np.reshape(netArr2,(-1,)),label='New')
            plt.yscale('symlog',linthreshy=0.01)
            plt.legend()
            plt.show()
            exit()
	
        # Throw the new tensors back in
        num = 0
        for n in net.network.nodes:
            for m in self.network.nodes:
                bidsn = list(b.id for b in n.buckets)
                bidsm = list(b.id for b in m.buckets)
                if len(set(bidsn).intersection(bidsm)) > 0:
                    print(bidsn, bidsm)
                    m.tensor = n.tensor
                    num += 1

        assert num == len(loop)

        # Verify error
        new = self.array

        print('---')
        err = np.sum((prev - new)**2) / np.sum(prev**2)


        if err > 3 * self.accuracy:

            ### There's a bug in loop optimizing which occasionally produces
            # considerably larger-than-expected accuracy.

            # Actually this might not be a bug, but rather a side effect
            # of the fact that in our procedure different links can end up
            # with very different sizes of terms on either side. One way
            # to address this is to apply a matrix and it's inverse between
            # each pair of in-loop and out-of-loop tensors such that each
            # link has comparable magnitude on the in-loop side. That is, we want
            # the matrix formed by contracting the loop against itself on all but
            # some given index to have norm comparable to that formed for any such index.
            # There's even more freedom, in fact, and in principle the transformation could
            # turn each such density matrix into the identity, though that'd be fiddly and
            # there's no reason to go so far. Instead we pick the matrix that first permutes
            # the matrix to have the largest elements on-diagonal and then normalise them
            # to unity.

            print(prev)
            print(new)
            print('___',err)
            print('___',err / self.accuracy)
            print(prevL)
            print(list([l.tensor.shape for l in loop]))
            import matplotlib.pyplot as plt
            plt.subplot(321)
            plt.plot(np.reshape(netArr,(-1,)),label='Original')
            plt.plot(np.reshape(netArr2,(-1,)),label='New')
            plt.yscale('symlog',linthreshy=0.01)
            plt.legend()
            plt.subplot(322)
            plt.plot(np.reshape((netArr-netArr2)/netArr,(-1,)),label='Residuals')
            plt.yscale('symlog',linthreshy=0.01)
            plt.subplot(323)
            plt.plot(np.reshape((netArr-netArr2)**2/np.sum(netArr**2),(-1,)),label='Error Contribution')
            plt.yscale('symlog',linthreshy=0.01)
            plt.subplot(324)
            plt.plot(np.reshape(prev,(-1,)),label='Original full')
            plt.plot(np.reshape(new,(-1,)),label='New full')
            plt.yscale('symlog',linthreshy=0.01)
            plt.legend()
            plt.subplot(325)
            plt.plot(np.reshape((prev-new)/prev,(-1,)),label='Residuals full')
            plt.yscale('symlog',linthreshy=0.01)
            plt.subplot(326)
            plt.plot(np.reshape((prev - new)**2/np.sum(prev**2),(-1,)),label='Error Contribution full')
            plt.yscale('symlog',linthreshy=0.01)
            plt.legend()
            plt.show()
            exit()

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
