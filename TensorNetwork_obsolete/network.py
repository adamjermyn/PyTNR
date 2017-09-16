from tensor import Tensor
from link import Link
from bucket import Bucket
from node import Node
import numpy as np
from compress import compress
from mergeLinks import mergeLinks
from priorityQueue import PriorityQueue


class Network:
    '''
    A Network is an object storing Nodes as well as providing helper methods
    for manipulating those Nodes. No Nodes in a Network may be ancestors of
    any other Nodes in that Network.
    '''

    def __init__(self, nodes):
        '''
        Method for initializing an empty Network.
        '''
        self._nodes = set(nodes)
        self._allLinks = set()

        for n in self._nodes:
            for b in n.buckets():
                if b.linked() and len(set(b.otherNodes()).intersection(self._nodes)) > 0:
                    self._allLinks.add(b.link())

    def __str__(self):
        '''
        Returns string representations of all top-level Nodes in the Network.
        The are no guarantees of the order in which this is done.
        '''
        s = 'Network\n'
        for n in self._nodes:
            s = s + str(n) + '\n'
        return s

    def size(self):
        '''
        Returns the sum of the sizes of all Tensors in the Network.
        '''
        return sum(n.tensor().size() for n in self._nodes)

    def nodes(self):
        '''
        Returns a copy of the set of all Nodes in the Network.
        '''
        return set(self._nodes)

    def representation(self):
        '''
        Returns the tensor product of all top-level Tensors along with
        a list of corresponding Buckets in the same order as the indices.
        '''
        arr = np.array([1.])
        logS = 0
        bucketList = []

        for n in self._nodes:
            arr = np.tensordot(arr, n.tensor().array(), axes=0)
            logS += n.logScalar()
            for b in n.buckets():
                bucketList.append(b)

        return arr[0], logS, bucketList

    def descend(self, node):
        '''
        This method replaces the specified Node with its children.
        This may entail descending other Nodes in the Network in
        order to maintain all Links.
        '''
        cc = node.children()

        if len(cc) == 0:
            return

        self._nodes.remove(node)

        for n in cc:
            self._nodes.add(n)

        for n in cc:
            descendSelf = False
            for b in n.buckets():
                if b.linked():
                    if len(set(b.otherNodes()).intersection(self._nodes)) == 0:
                        intersection = b.otherTopNode().ancestors().intersection(self._nodes)
                        if len(intersection) == 0:
                            descendSelf = True
                        else:
                            nn = intersection.pop()
                            self.descend(nn)
            if descendSelf:
                self.descend(n)
                return

    def ascend(self, node):
        '''
        This method replaces the specified Node with its parent.
        This may entail ascending other Nodes in the Network (and
        removing some) in order ot maintain all Links as well as
        the property that no Node be an ancestor of any other Node.
        '''
        p = node.parent()

        if p is None:
            return

        self._nodes.add(p)

        for c in p.children():
            self._nodes.remove(c)

        for b in p.buckets():
            while len(set(b.otherNodes()).intersection(self._nodes)) == 0:
                intersection = b.otherTopNode().allNChildren().intersection(self._nodes)
                assert len(intersection) == 1
                nn = intersection.pop()
                self.ascend(nn)
