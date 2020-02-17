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

    def __deepcopy__(self, memodict={}):
        new = type(self)(self.accuracy)
        new.network = deepcopy(self.network)
        
        idDict = {}
        for n in new.network.nodes:
            for b in n.buckets:
                idDict[b.id] = b
        
        for b in self.externalBuckets:
            new.externalBuckets.append(idDict[b.id])

        return new        

    def newIDs(self):
        nidDict = {}
        bidDict = {}
        for n in self.network.nodes:
            nid = n.id
            n.id = Node.newid()
            nidDict[nid] = n.id
        for b in self.network.buckets:
            bid = b.id
            b.id = Bucket.newid()
            bidDict[bid] = b.id
        return nidDict, bidDict
    
    def copy(self):
        new = deepcopy(self)
        nidDict, bidDict = new.newIDs()
        for b in new.externalBuckets:
            b.link = None
        return new, nidDict, bidDict

    def addTensor(self, tensor):
        n = Node(tensor, Buckets=[Bucket() for _ in range(tensor.rank)])
        self.network.addNode(n)
        self.externalBuckets.extend(n.buckets)
        if tensor.rank > 3:
            self.network.splitNode(n)
        return n

    def addNode(self, node):
        assert node in self.network.nodes

        # The external buckets associated this node are added at the end of the list.
        for b in node.buckets:
            if not b.linked or b.otherBucket not in self.externalBuckets:
                self.externalBuckets.append(b)
            elif b.linked and b.otherBucket in self.externalBuckets:
                self.externalBuckets.remove(b.otherBucket)

        self.network.addNode(node)


    def removeNode(self, node):
        assert node in self.network.nodes

        # The external buckets associated with the bonds broken by
        # removing this node are added at the end of the list.
        for b in node.buckets:
            if b not in self.externalBuckets:
                self.externalBuckets.append(b.otherBucket)
            else:
                self.externalBuckets.remove(b)

        self.network.removeNode(node)

        # Now we de-register all links to the removed node.
        # This is not handled internally by the network because doing so
        # interferes with the contraction process. TODO: Refactor so that the following
        # lines are not needed.
        for b in node.buckets:
            if b.link is not None:
                b.otherBucket.link = None


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
    def logNorm(self):
        '''
        :return: The log of the square root of the sum of squared entries in the tensor.
        '''
        arr, logAcc, bdict = self.network.array
        logArr = 0.5 * np.log(np.sum(arr**2))
        return logArr + logAcc

  
    @property
    def disjointArrays(self):
        '''
        Calculates an array for each disjoint subgraph of the underlying Network.
        
        :return: List of arrays and list of corresponding lists of external Buckets.
        '''
        
        nets = self.network.disjointNetworks()

        arrs = []
        buckets = []
        logs = []
        
        for n in nets:
            arr, logAcc, bdict = n.array

            perm = []
            blist = [b.id for b in self.externalBuckets if b.id in bdict.keys()]

            for b in blist:
                perm.append(bdict[b])
            
            arr = np.transpose(arr, axes=perm)
            
            arrs.append(arr)
            buckets.append(blist)
            logs.append(logAcc)

        return arrs, buckets, logs

    @property
    def scaledArray(self):

        arr, logAcc, bdict = self.network.array

        perm = []
        blist = [b.id for b in self.externalBuckets]

        for b in blist:
            perm.append(bdict[b])

        arr = np.transpose(arr, axes=perm)

        assert arr.shape == tuple(self.shape)
        return arr

    @property
    def logScalar(self):
        return np.sum(n.tensor.logScalar for n in self.network.nodes)


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

    def contractToArray(self, ind, other, otherInd):
        '''
        Contracts self with other and returns an array without constructing intermediate NetworkTensor objects.
        Always equivalent to front=True.
        '''
        
        return self.contract(ind, other, otherInd).array
        
    def contract(self, ind, other, otherInd, front=True, elimLoops=False):
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
        self.network.contractRank2()

