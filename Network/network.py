from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.compress import compressLink

from copy import deepcopy
import numpy as np
import networkx

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['network'])


class Network:
    '''
    A Network is an object storing Nodes as well as providing helper methods
    for manipulating those Nodes.
    '''

    def __init__(self):
        self.nodes = set()
        self.buckets = set()
        self.internalBuckets = set()
        self.externalBuckets = set()
        self.optimizedLinks = set()

    def __str__(self):
        s = 'Network\n'
        for n in self.nodes:
            s = s + str(n) + '\n'
        return s

    @property
    def array(self):
        '''
        Contracts the network down to an array object.
        Indices are ordered by ascending bucket ID.
        Returns the array, the log of a prefactor, and a bucket dictionary.
        '''
        net = deepcopy(self)

        isolated = set()

        ret = []
        while len(net.nodes) > 0:
            n = net.nodes.pop()
            net.nodes.add(n)

            c = net.internalConnected(n)
            if len(c) > 0:
                c = c.pop()
                net.mergeNodes(n, c)
            else:
                # Means that n is an isolated node
                isolated.add(n)
                net.nodes.remove(n)

        arr = 1
        logAcc = 0
        buckets = []
        while len(isolated) > 0:
            n = isolated.pop()
            arr2 = n.tensor.array
            logAcc += np.log(np.max(np.abs(arr2)))
            arr = np.tensordot(arr, arr2 / np.max(np.abs(arr2)), axes=0)
            logger.debug('Computing array. Log is ' + str(logAcc) + '.')
            buckets.extend(n.buckets)

        bids = [b.id for b in buckets]
        bids = sorted(bids)

        bdict = {}
        for i in range(len(buckets)):
            bdict[buckets[i].id] = i

        perm = []
        for b in bids:
            perm.append(bdict[b])

        if len(perm) > 0:
            arr = np.transpose(arr, axes=perm)

        bdict = {}
        for i in range(len(buckets)):
            bdict[bids[i]] = i

        return arr, logAcc, bdict

    def addNode(self, node):
        '''
        Registers a new Node in the Network.
        This should only be called when registering a new Node.
        All links between this node and other nodes in this network
        must already exist, so in that sense adding the Node ought to
        be the last thing that is done.
        '''
        assert node not in self.nodes
        assert node.network is None

        node.network = self
        self.nodes.add(node)
        for b in node.buckets:
            self.buckets.add(b)
            if b.linked and b.otherNode in self.nodes:
                self.internalBuckets.add(b)
                self.internalBuckets.add(b.otherBucket)
                if b.otherBucket in self.externalBuckets:
                    self.externalBuckets.remove(b.otherBucket)
            else:
                self.externalBuckets.add(b)

    def removeNode(self, node):
        '''
        De-registers a Node from the Network.
        This should only be called when deleting a Node.
        This also handles updating the link registration
        in the event that the Node was formed from contracting
        a Link.
        '''
        assert node in self.nodes

        node.network = None
        self.nodes.remove(node)
        for b in node.buckets:
            self.buckets.remove(b)
            if b in self.internalBuckets:
                self.internalBuckets.remove(b)
                if b.otherBucket in self.internalBuckets:
                    self.internalBuckets.remove(b.otherBucket)
                    self.externalBuckets.add(b.otherBucket)
            if b in self.externalBuckets:
                self.externalBuckets.remove(b)

    def dummyMergeNodes(self, n1, n2):
        '''
        Calculates the tensor and bucket array which would arise
        were the nodes n1 and n2 to be merged.
        '''
        links = n1.linksConnecting(n2)
        indices = [[], []]
        for l in links:
            b1, b2 = l.bucket1, l.bucket2
            if b1.node == n2:
                b1, b2 = b2, b1
            assert b1.node == n1
            assert b2.node == n2
            assert b1.otherBucket == b2
            assert b2.otherBucket == b1
            indices[0].append(b1.index)
            indices[1].append(b2.index)

        t = n1.tensor.contract(indices[0], n2.tensor, indices[1])

        buckets = []
        for b in n1.buckets:
            if not b.linked or b.otherBucket not in n2.buckets:
                buckets.append(b)
        for b in n2.buckets:
            if not b.linked or b.otherBucket not in n1.buckets:
                buckets.append(b)

        return t, buckets

    def mergeNodes(self, n1, n2):
        '''
        Merges the specified Nodes.
        '''
        t, buckets = self.dummyMergeNodes(n1, n2)

        n = Node(t, Buckets=buckets)

        # The order matters here: we have to remove the old nodes before
        # adding the new one to make sure that the correct buckets end up
        # in the network.
        self.removeNode(n1)
        self.removeNode(n2)
        self.addNode(n)

        return n

    def mergeLinks(self, n, compress=False, accuracy=1e-4):
        merged = []
        for n1 in n.connectedNodes:
            links = n1.linksConnecting(n)
            buckets1 = []
            buckets2 = []
            if len(links) > 1:
                for l in links:
                    if l.bucket1.node is n:
                        buckets1.append(l.bucket1)
                        buckets2.append(l.bucket2)
                    else:
                        buckets1.append(l.bucket2)
                        buckets2.append(l.bucket1)
                b = n.mergeBuckets(buckets1)
                b1 = n1.mergeBuckets(buckets2)
                l = Link(b, b1)
                if compress:
                    compressLink(l, accuracy)
                merged.append(n1)
        return merged

    def mergeClosestLinks(self, n1, compress=False, accuracy=1e-4):
        best = [1e100, None, None, None, None]

        for n2 in n1.connectedNodes:
            if hasattr(n2.tensor, 'compressedSize'):
                links = n1.linksConnecting(n2)
                buckets1 = []
                buckets2 = []
                if len(links) > 1:
                    for l in links:
                        if l.bucket1.node is n1:
                            buckets1.append(l.bucket1)
                            buckets2.append(l.bucket2)
                        else:
                            buckets1.append(l.bucket2)
                            buckets2.append(l.bucket1)
                    for b11 in buckets1:
                        for b12 in buckets1:
                            if b11 != b12:
                                dist = n1.tensor.distBetweenBuckets(b11, b12)
                                dist += n2.tensor.distBetweenBuckets(
                                    b11.otherBucket, b12.otherBucket)
                                dist *= np.log(b11.size * b12.size)
                                if dist < best[0]:
                                    best = [
                                        dist, n2, b11, b12, b11.otherBucket, b12.otherBucket]

        if best[0] < 1e100:
            dist, n2, b11, b12, b21, b22 = best

            b = n1.mergeBuckets([b11, b12])
            b1 = n2.mergeBuckets([b21, b22])
            l = Link(b, b1)
            if compress:
                compressLink(l, accuracy)

            return n2

        return None

    def internalConnected(self, node):
        return self.nodes.intersection(set(node.connectedNodes))

    def toGraph(self):
        g = networkx.Graph()
        g.add_nodes_from(self.nodes)
        for n in self.nodes:
            for m in self.internalConnected(n):
                g.add_edge(n, m, weight=np.log(n.tensor.size * m.tensor.size))
        return g
