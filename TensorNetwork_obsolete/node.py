from link import Link
from bucket import Bucket
from tensor import Tensor
from mergeLinks import mergeLinks
from collections import Counter
import numpy as np


class Node:
    '''
    A Node object stores the Buckets corresponding to a given Tensor.
    It also provides helper methods allowing for easy access to neighbors.

    Node objects have the following functions which do not modify them:

    id 				-	Returns the id number of the Node. These numbers are unique within a network.
    children		-	Returns the children Nodes (if any) which merged to form this Node.
    parent			-	Returns the Node (if any) which this merges to form.
    topParent		-	Returns the highest-level ancestor of this Node.
    allNChildren	-	Returns all Nodes for which topParent returns this Node.
    network 		-	Returns the Network this Node belongs to.
    connected		-	Returns the Nodes this one is connected to.
    connectedHigh	-	Returns the Nodes this one is connected to, giving the highest-level list possible.
                                            That is, the list such that no element of the set has a parent which is connected
                                            to this Node, and such that all Nodes connected to this one are children to some
                                            degree of an element in the list.
    tensor 			-	Returns the Tensor underlying this Node.
    logScalar 		-	Returns the log of the scalar component that has been divided out from this Tensor.
    bucket 			-	Returns the Bucket at the given index.
    buckets 		-	Returns all Buckets.
    bucketIndex 	-	Returns the index of the specified bucket.
    findLink		-	Takes as input another Node and finds the Link between this Node that,
                                            if one exists. If none exists returns None.
    linksConnecting	-	Returns all Links connecting this Node and another (provided as input).

    There are functions which modify Nodes by linking them or by setting heirarchy attributes:

    addLink			-	Takes as input another Node as well as the index of the Bucket on this Node
                                            and the index of the Bucket on the other Node. Links them. If they are already
                                            Linked this raises a ValueError.

    setParent		-	Sets the parent of this Node to the reference given.

    There are additional functions which create modified copies of Nodes, listed below.
    These may be called only if the node is parentless.

    modify			-	Creates a copy of this Node with the provided Tensor instead. The copy is
                                            one level up in the heirarchy, such that the copy is the parent of this Node.
                                            This also properly links the copy to the rest of the Network.
    trace			-	Searches for indices which are linked to one another and produces a new Node
                                            with them traced out. The new Node is then the parent of this Node.
    merge 			-	Takes as input another Node and merges this Node with it. The net result is that
                                            a new node is added to the network. This Node and the other are left intact, with
                                            all of their Links preserved. The rest of the network's Nodes will have Links both
                                            to these and to the new Node, with Links to the new Node appearing later in their
                                            Link lists.
    linkMerge		-	Searches for linked Nodes at the top level which have multiple Links between them and
                                            this one and produces a new pair of nodes as their parents with the Links merged into
                                            a single higher-dimensional Link.


    Finally, nodes may be deleted:

    delete			-	Delete the Node and all associated Links. Recursively deletes all parents.
    '''

    def __init__(
            self,
            tens,
            network,
            children=None,
            Buckets=None,
            logScalar=0):
        self._tensor = tens
        self._logScalar = logScalar + self._tensor.logScalar()
        self._network = network
        self._id = self._network.nextID()
        self._parent = None

        if children is None:
            children = []
        if Buckets is None:
            Buckets = []

        self._children = children
        self._buckets = Buckets
        self._network.registerNode(self)
        for b in Buckets:
            b.addNode(self)
        for c in self._children:
            c.setParent(self)

    def id(self):
        return self._id

    def __str__(self):
        return 'Node with ID: ' + \
            str(self._id) + '  and tensor shape ' + str(self._tensor.shape())

    def children(self):
        return self._children

    def parent(self):
        return self._parent

    def topParent(self):
        if self._parent is None:
            return self
        else:
            return self.parent().topParent()

    def bottomChild(self):
        if len(self._children) == 0:
            return self
        else:
            return self._children[0]

    def sibling(self):
        if self._parent is None:
            return None
        else:
            sibs = self._parent.children()
            if len(sibs) == 2:
                if sibs[0] == self:
                    return sibs[1]
                else:
                    return sibs[0]

    def allNChildren(self):
        ch = set(self._children)
        for c in self._children:
            ch = ch | set(c.allNChildren())
        return ch

    def ancestors(self):
        anc = set()
        n = self
        while n._parent is not None:
            anc.add(n._parent)
            n = n._parent
        return anc

    def setParent(self, parent):
        self._parent = parent

    def network(self):
        return self._network

    def connected(self):
        c = []
        for b in self._buckets:
            if b.linked():
                c.extend(b.otherNodes())
        return c

    def connectedHigh(self):
        c = []
        for b in self._buckets:
            if b.linked():
                c.append(b.otherTopNode())
        return c

    def findLink(self, other):
        for b in self._buckets:
            if b.linked():
                if other in b.otherNodes():
                    return b.link()
        return None

    def linksConnecting(self, other):
        links = []
        for b in self._buckets:
            if b.linked():
                if other in b.otherNodes():
                    links.append(b.link())
        return links

    def indexConnecting(self, other):
        for i, b in enumerate(self._buckets):
            if b.linked():
                if other in b.otherNodes():
                    return i
        return None

    def tensor(self):
        return self._tensor

    def logScalar(self):
        return self._logScalar

    def bucketIndex(self, b):
        return self._buckets.index(b)

    def bucket(self, i):
        return self._buckets[i]

    def buckets(self):
        return self._buckets

    def removeBucket(self, b):
        assert b in self._buckets
        self._buckets.remove(b)

    def delete(self, linksToDelete=None):
        # Delete parents
        if self._parent is not None:
            self._parent.delete()

        assert self._parent is None

        # Delete buckets whose links ought to be deleted,
        # and keep track of those links.
        if linksToDelete is None:
            linksToDelete = []

        for b in self._buckets:
            if b.linked():
                link = b.link()
                bo = link.otherBucket(b)
                numC = len(link.children())
                if b.numNodes() == 1 and numC > 0:
                    # Means we're about to delete a link which is compressed or
                    # merged.
                    no = b.otherBottomNode()
                    bo = link.otherBucket(b)

                    if no.parent() is not None:
                        no.parent().delete()

                    # Now the other Node has no parents, so we've got a simple case
                    # of two top-level Nodes on either side of a link, each with buckets that
                    # have one node a piece.

                    self.removeBucket(b)
                    no.removeBucket(bo)

                    linksToDelete.append(link)

                    # The order here matters: whichever we delete last has to get the
                    # list of links to be deleted, so that we can delete both nodes before
                    # we delete the link.
                    self.delete()
                    no.delete(linksToDelete=linksToDelete)
                    return

        # At this stage we have no buckets pointing to links which were
        # compressed or merged.
        for b in self._buckets:
            if b.linked():
                assert len(b.link().children()) == 0 or b.numNodes() > 1

        # We deregister the node before we handle any link deletion.
        self._network.deregisterNode(self)
        assert self not in self._network.nodes()
        assert self not in self._network.topLevelNodes()
        for c in self.children():
            assert c in self._network.topLevelNodes()

        # Now we delete the links
        for link in linksToDelete:
            assert self in [link.bucket1().topNode(), link.bucket2().topNode()]
            assert len(set([link.bucket1().topNode(), link.bucket2().topNode()]).intersection(
                self._network.topLevelNodes())) == 0
            assert link.parent() is None or link.parent() == link
            link.delete()

        # Note that we don't need to delete the buckets associated with the Links
        # we deleted, as those Links no longer refer to them and the Nodes no longer
        # refer to them either.

        for b in self._buckets:
            b.removeNode()

        for c in self.children():
            c.setParent(None)

    def addLink(
            self,
            other,
            selfBucketIndex,
            otherBucketIndex,
            optimized=False,
            compressed=False,
            children=None):
        assert self in self._network.topLevelNodes()
        assert other in self._network.topLevelNodes()
        assert children is None or len(
            self.children()) == len(
            other.children())
        assert self.tensor().shape()[selfBucketIndex] > 1

        if children is None:
            children = []

        selfBucket = self.bucket(selfBucketIndex)
        otherBucket = other.bucket(otherBucketIndex)

        if selfBucket.linked():
            raise ValueError
        if otherBucket.linked():
            raise ValueError

        l = Link(
            selfBucket,
            otherBucket,
            self._network,
            compressed=compressed,
            optimized=optimized,
            children=children)

        selfBucket.setLink(l)
        otherBucket.setLink(l)

        return l

    def modify(self, tens, delBuckets=None, repBuckets=None):
        '''
        len(delBuckets) + len(tens.shape()) - len(newBuckets) == len(self.tensor().shape())
        Creates a copy of this Node with tens as its Tensor.  Omits buckets at indices listed in
        delBuckets. Replaces Buckets at indices listed in repBuckets with new Bucket objects.
        Raises a ValueError if repBuckets and delBuckets contain overlapping elements.
        '''
        if delBuckets is None:
            delBuckets = []

        if repBuckets is None:
            repBuckets = []

        assert self in self._network.topLevelNodes()
        assert len(set(delBuckets).intersection(set(repBuckets))) == 0
        assert len(delBuckets) + len(tens.shape()) - \
            len(self.tensor().shape()) >= 0

        Buckets = []

        for i, b in enumerate(self.buckets()):
            if i not in delBuckets:
                if i not in repBuckets:
                    Buckets.append(b)
                else:
                    Buckets.append(Bucket(self.network()))

        n = Node(
            tens,
            self._network,
            children=[self],
            Buckets=Buckets,
            logScalar=self._logScalar)

        return n

    def trace(self):
        assert self in self._network.topLevelNodes()

        # Find all self-Links

        axes0 = []
        axes1 = []
        links = []

        for b in self._buckets:
            if b.linked():
                otherBucket = b.otherBucket()
                otherNode = otherBucket.topNode()
                if otherNode == self:
                    ind0 = self.bucketIndex(b)
                    ind1 = self.bucketIndex(otherBucket)

                    # Avoid duplicates
                    if ind0 not in axes1:
                        assert ind1 not in axes0
                        links.append(b.link())
                        axes0.append(ind0)
                        axes1.append(ind1)

        # Trace

        if len(axes0) > 0:
            for l in links:
                self._network.registerLinkCut(l)
            newT = self._tensor.trace(axes0, axes1)
            n = self.modify(newT, delBuckets=(axes0 + axes1))
            return n
        else:
            return self

    def linkMerge(self, compressL=False, eps=1e-4):
        assert self._parent is None

        todo = set()
        done = set()
        new = set()

        c = Counter(self.connectedHigh())

        for n in c:
            if c[n] > 1:
                todo.add(n)

        n1 = self

        while len(todo) > 0:
            n = todo.pop()
            done.add(n)

            n1, n2, _ = mergeLinks(n1, n, compressLink=compressL)

            assert len(n1.children()) == len(n2.children())

            new.add(n2)

        return n1, done, new

    def merge(self, other, mergeL=True, compressL=True, eps=1e-4):
        assert self in self._network.topLevelNodes()
        assert other in self._network.topLevelNodes()
        assert self in other.connectedHigh()
        assert other in self.connectedHigh()
        assert self != other

        # Find all links between self and other, store their indices, and
        # deregister them from the top level
        links = []
        for i, b in enumerate(self._buckets):
            if b.linked():
                if b.otherTopNode() == other:
                    links.append((i, other.bucketIndex(b.otherBucket())))
                    self._network.deregisterLinkTop(b.link())

        links = list(zip(*links))

        # Contract along common links
        t = self._tensor.contract(links[0], other.tensor(), links[1])

        # Build new Node
        Buckets = []
        for b in self.buckets():
            if not b.linked():
                Buckets.append(b)
            elif b.otherTopNode() != other:
                Buckets.append(b)
        for b in other.buckets():
            if not b.linked():
                Buckets.append(b)
            elif b.otherTopNode() != self:
                Buckets.append(b)

        # Build new Node
        n = Node(
            t,
            self._network,
            children=[
                self,
                other],
            Buckets=Buckets,
            logScalar=self.logScalar() +
            other.logScalar())

        # Trace out any self-loops
        n = n.trace()

        if mergeL:
            # Merge any links that need it. The next line is probably redundant
            # given that trace now returns the top level.
            n = n.topParent()
            n, _, _ = n.linkMerge(compressL=compressL, eps=eps)

        return n
