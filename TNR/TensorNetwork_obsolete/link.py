import numpy as np
from tensor import Tensor


class Link:
    '''
    A Link is a means of indicating an intent to contract two tensors.
    This object has two Buckets, one for each Node being connected.
    In addition, it has a method for computing the von Neumann entropy of the Link.
    In cases where this computation is intractable due to memory requirements, a heuristic
    is used.

    Links have the following functions:

    bucket1			-	Returns the first Bucket this Link connects to.
    bucket2			-	Returns the second Bucket this Link connects to.
    network 		-	Returns the Network this link belongs to.
    parent 			-	Returns the parent of this Link.
    children 		-	Returns the children of this Link.
    setParent		-	Takes as input a Link and sets it as the parent of this Link.
    otherBucket		-	Takes as input a Bucket. Raises a ValueError if it is not one of the
                                            Buckets associated with this Link. If it is one of them, returns the other.
    mergeEntropy	-	Returns the expected change in the entropy of the network
                                            were the link to be contracted. Heuristics are used on the
                                            assumption that links are being compressed regularly. Assumes compression
                                            would be performed on top-Level nodes.
    updateMergeEntropy	-	Updates the stored merge entropy. Should only be called from the update() method.
    update 			-	Called whenever a Bucket this Link points to gains or loses a Node. Updates merged
                                            entropy and corrects/registers the top/not top status of the Link.
    compressed 		-	Returns True if this Link was the result of a compression operation, False otherwise.
    setCompressed	-	Sets the Link compress variable to True.
    delete			-	Removes this link from both

    Links are instantiated with the buckets they connect, and are added to the end of the Link
    lists of their buckets. They are also added to the link registry of their TensorNetwork.
    '''

    def __init__(
            self,
            b1,
            b2,
            network,
            compressed=False,
            optimized=False,
            reduction=0.75,
            children=None):
        if children is None:
            children = []
        self._b1 = b1
        self._b2 = b2
        self._compressed = compressed
        self._network = network
        self._reduction = reduction
        self._mergeEntropy = None
        self._parent = None
        self._children = children
        self._optimized = optimized
        for c in self._children:
            c.setParent(self)
            self._network.deregisterLinkTop(c)
        self._network.registerLink(self)

    def bucket1(self):
        return self._b1

    def bucket2(self):
        return self._b2

    def network(self):
        return self._network

    def parent(self):
        return self._parent

    def setParent(self, parent):
        self._parent = parent

    def children(self):
        return self._children

    def compressed(self):
        return self._compressed

    def setCompressed(self):
        self._compressed = True

    def topContents(self):
        n1 = self._b1.topNode()
        n2 = self._b2.topNode()

        t1 = n1.tensor()
        t2 = n2.tensor()

        sh1 = t1.shape()
        sh2 = t2.shape()

        return n1, n2, t1, t2, sh1, sh2

    def otherBucket(self, bucket):
        if bucket == self._b1:
            return self._b2
        elif bucket == self._b2:
            return self._b1
        else:
            raise ValueError

    def optimized(self):
        return self._optimized

    def setOptimized(self):
        self._optimized = True

    def mergeEntropy(self):
        if self._mergeEntropy is None:
            self.updateMergeEntropy()
        return self._mergeEntropy

    def updateMergeEntropy(self):
        assert self in self._network.topLevelLinks()

        # Entropy is computed in base e.
        # As a heuristic we assume that merging a bond of
        # size S with a bond of size S' produces a bond of
        # size reduction*S*S'.

        n1 = self._b1.topNode()
        n2 = self._b2.topNode()

        t1 = n1.tensor()
        t2 = n2.tensor()

        length = t1.shape()[n1.bucketIndex(self._b1)]

        if n1 == n2:
            self._mergeEntropy = t1.size() / (length**2) - t1.size()
        else:
            s1 = t1.size()
            s2 = t2.size()

            sN = s1 * s2 / length**2  # Estimate based on no merging of Links

            # Correction based on number of merging Links
            commonNodes = set(n1.connectedHigh()).intersection(
                set(n2.connectedHigh()))
            sN *= self._reduction**len(commonNodes)

            dS = sN - s1 - s2

            self._mergeEntropy = dS

    def update(self):
        self.updateMergeEntropy()
        if self._parent is None:
            # I have a feeling there's a more elegant way to handle this, but
            # I'm not sure what it is.
            self._network.updateSortedLinkList(self)

    def delete(self):
        # This method assumes you want to delete the nodes on either side!
        assert len(self._children) > 0

        self._network.deregisterLink(self)
        self._b1.removeLink()
        self._b2.removeLink()

        # This means that if we remove the references to these buckets in the Nodes then
        # the buckets will be deleted (there will be no remaining references to them
        # assuming they are in fact only referenced by a single Node each).
        self._b1 = None
        self._b2 = None

        for c in self._children:
            c.setParent(None)
            self._network.registerLinkTop(c)
