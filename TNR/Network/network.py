from TNR.Network.bucket import Bucket
from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.compress import compressLink
from TNR.Tensor.arrayTensor import ArrayTensor
from copy import deepcopy
import numpy as np
import networkx

from opt_einsum import contract as einsum

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

        # A placeholder for the networkx graph representation.
        # This can be maintained iteratively, but if it is ever
        # set to None we just construct it from scratch when it is
        # next requested.
        self.graph = None

    def __str__(self):
        s = 'Network\n'
        for n in self.nodes:
            s = s + str(n) + '\n'
        return s

    def copy(self):
        new = deepcopy(self)
        for n in new.nodes:
            n.id = Node.newid()
        for b in new.buckets:
            b.id = Bucket.newid()
        for b in new.externalBuckets:
            b.link = None
        return new

    def __deepcopy__(self, memodict={}):
        # Deep copy

        nodes = list(self.nodes)

        # Copy nodes
        newNodes = []
        for n in nodes:
            buckets = []
            for b in n.buckets:
                buckets.append(Bucket())
                buckets[-1].id = b.id
            n2 = Node(deepcopy(n.tensor), Buckets=buckets)
            n2.id = n.id
            newNodes.append(n2)
        
        # Create links
        for j,n in enumerate(nodes):
            for i,b in enumerate(n.buckets):
                if b.linked and not newNodes[j].buckets[i].linked:                   
                    otherNode = newNodes[nodes.index(b.otherBucket.node)]
                    otherInd = b.otherBucket.node.buckets.index(b.otherBucket)
                    l = Link(newNodes[j].buckets[i], otherNode.buckets[otherInd])
                    l.id = b.link.id

        # Add nodes
        new = type(self)()
        for n in newNodes:
            new.addNode(n)
                        
        return new

    @property
    def array(self):
        '''
        Contracts the network down to an array object.
        Indices are ordered by ascending bucket ID.
        Returns the array, the log of a prefactor, and a bucket dictionary.
        Uses numpy's einsum feature, with order optimization.
        '''
                
                
        net = deepcopy(self)
        net.cutLinks()
        net.contractRank2()
        
        
        # Fix node order
        nodes = list(net.nodes)

        # Setup subscript arrays
        subs = list([-1 for _ in range(len(n.buckets))] for n in nodes)
        out = []
        bids = []

        # Construct lists
        counter = 0
        for i in range(len(nodes)):
            n = nodes[i]

            for j,b in enumerate(n.buckets):
                if subs[i][j] == -1:
                    subs[i][j] = counter
                    if b.linked:
                        ind = nodes.index(b.otherBucket.node)
                        ind2 = nodes[ind].buckets.index(b.otherBucket)
                        subs[ind][ind2] = counter
                    else:
                        out.append(counter)
                        bids.append(b.id)
                    counter += 1
    
        args = []
        for i in range(len(nodes)):
            args.append(nodes[i].tensor.scaledArray)
            args.append(subs[i])
        args.append(out)

        arr = einsum(*args, optimize='greedy', memory_limit=1e7)

        logAcc = sum(n.tensor.logScalar for n in nodes)

        bdict = {}
        for i in range(len(bids)):
            bdict[bids[i]] = i

        bids = sorted(bids)

        perm = []
        for b in bids:
            perm.append(bdict[b])

        if len(perm) > 0:
            arr = np.transpose(arr, axes=perm)


        bdict = {}
        for i in range(len(bids)):
            bdict[bids[i]] = i

        return arr, logAcc, bdict

    def copySubset(self, nodes):
        '''
        Produces a deepcopy (ID-preserving) of self with only the specified Nodes included.
        
        :param nodes: The Nodes to retain.
        :return: Network containing only the specified Nodes.
        '''
        net = deepcopy(self)

        # Prune the network down to just the specified nodes
        ids = list([n.id for n in nodes])
        nodes2 = list(net.nodes)

        for n in nodes2:
            if n.id not in ids:
                net.removeNode(n)

        return net

    def disjointNetworks(self):
        '''
        Splits the Network into its minimal disjoint subgraphs (i.e. connected components).
        Note: These are constructed using deepcopy so that ID's are preserved.
        
        :return: List of Networks each of which contains one connected component.
        '''
                
        g = self.toGraph()
        components = list(networkx.connected_components(g))

        nets = list(self.copySubset(component) for component in components)        
        
        return nets

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

        if self.graph is not None:
            self.graph.add_node(node)

        node.network = self
        self.nodes.add(node)
        for b in node.buckets:
            self.buckets.add(b)
            if b.linked and b.otherNode in self.nodes:
                self.internalBuckets.add(b)
                self.internalBuckets.add(b.otherBucket)
                if self.graph is not None:
                    self.graph.add_edge(node, b.otherNode)
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

        if self.graph is not None:
            self.graph.remove_node(node)

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

    def removeLink(self, link):
        '''
        Deletes a link between two internal nodes.
        Updates registration of internal and external buckets accordingly.
        '''

        b1, b2 = link.bucket1, link.bucket2
        n1, n2 = b1.node, b2.node

        if self.graph is not None:
            self.graph.remove_edge(n1, n2)

        b1.link = None
        b2.link = None

        self.internalBuckets.remove(b1)
        self.internalBuckets.remove(b2)

        self.externalBuckets.add(b1)
        self.externalBuckets.add(b2)


    def check(self):
        '''
        Checks that all links in the network are valid.
        '''
        for n1 in self.nodes:
            for n2 in self.internalConnected(n1):
                links = n1.linksConnecting(n2)
                for l in links:
                    b1, b2 = l.bucket1, l.bucket2
                    if b1.node == n2:
                        b1, b2 = b2, b1
                    assert b1.node == n1
                    assert b2.node == n2
                    assert b1.otherBucket == b2
                    assert b2.otherBucket == b1
                    assert b1.size == b2.size

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
            assert b1.size == b2.size
            indices[0].append(b1.index)
            indices[1].append(b2.index)

        if hasattr(n1.tensor, 'artificialCut'):
            t = n1.tensor.contract(indices[0], n2.tensor, indices[1], elimLoops=False)
        else:
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

    def traceOut(self, b):
        '''
        Traces out the component of the tensor associated with the bucket b.
        '''

        assert b in self.externalBuckets

        n = b.node
        n.tensor = n.tensor.traceOut(b.index)
        n.buckets.remove(b)
        self.externalBuckets.remove(b)

    def internalConnected(self, node):
        return self.nodes.intersection(set(node.connectedNodes))

    def toGraph(self):
        if self.graph is None:
            g = networkx.Graph()
            g.add_nodes_from(self.nodes)
            for n in self.nodes:
                for m in self.internalConnected(n):
                    g.add_edge(n, m, weight=np.log(n.tensor.size * m.tensor.size))
            self.graph = g

        return self.graph

    def contractRank2(self):
        done = set()
        while len(
            done.intersection(
                self.nodes)) < len(
                self.nodes):

            n = next(iter(self.nodes.difference(done)))

            nodes = self.internalConnected(n)
            if len(nodes) == 0:
                done.add(n)
            elif n.tensor.rank <= 2:
                self.mergeNodes(n, nodes.pop())
            else:
                merged = False
                for n2 in nodes:
                    if n.tensor.rank + n2.tensor.rank - 2*len(n.findLinks(n2)) <= 3:
                        self.mergeNodes(n, n2)
                        merged = True
                        break
                if not merged:
                    done.add(n)


    def cutLinks(self):
        '''
        Identifies links with dimension 1 and eliminates them.
        '''

        for n in self.nodes:
            for m in self.internalConnected(n):
                dim = 1
                while dim == 1 and m in self.internalConnected(n):
                    inds = n.indicesConnecting(m)
                    i = inds[0][0]
                    j = inds[1][0]
                    dim = n.tensor.shape[i]
                    if dim == 1:
                        sl = list(slice(0,n.tensor.shape[k]) for k in range(i)) + [0] + list(slice(0,n.tensor.shape[k]) for k in range(i+1,n.tensor.rank))
                        n.tensor = ArrayTensor(n.tensor.array[sl])
                        self.internalBuckets.remove(n.buckets[i])
                        n.buckets.remove(n.buckets[i])
                        sl = list(slice(0,m.tensor.shape[k]) for k in range(j)) + [0] + list(slice(0,m.tensor.shape[k]) for k in range(j+1,m.tensor.rank))
                        m.tensor = ArrayTensor(m.tensor.array[sl])
                        self.internalBuckets.remove(m.buckets[j])
                        m.buckets.remove(m.buckets[j])

        for n in self.nodes:
            for b in n.buckets:
                if b in self.internalBuckets:
                    assert b.size > 1

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