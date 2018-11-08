import numpy as np
from copy import deepcopy
from TNR.Network.link import Link
from TNR.Network.node import Node
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Utilities import svd
from TNR.Utilities.linalg import L2error
from TNR import config

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])


def trivialCut(loop, link):
    '''
    Cuts the loop by running the specified index through a series of
    identity tensors.
    :param loop: NetworkTensor specifying a loop.
    :param link: The link to cut. 
    :return: A cut NetworkTensor which is exactly equal to loop upon contraction.
    '''

    # Identify bucket and node indices of link
    bid1 = link.bucket1.id
    bid2 = link.bucket2.id
    nid1 = link.bucket1.node.id
    nid2 = link.bucket2.node.id

    # Copy tensor
    loop2 = deepcopy(loop)

    # Identify copied nodes and buckets
    for n in loop2.network.nodes:
        if n.id == nid1:
            n1 = n
        elif n.id == nid2:
            n2 = n

    for b in n1.buckets:
        if b.id == bid1:
            b1 = b
            
    for b in n2.buckets:
        if b.id == bid2:
            b2 = b
            
    # Construct identity tensors
    idenNodes = [] # Holds identity nodes
    idenBuckets = [] # Holds buckets to contract with identities.
    nodes = list(loop2.network.nodes)
    for n in nodes:
        if n != n1 and n != n2:
            iden = ArrayTensor(np.identity(b1.size))
            idenNode = Node(iden)
            idenNodes.append(idenNode)
            idenBuckets.append(list(idenNode.buckets))

    # Because we'll be linking them with the identity
    # we add b1 and b2 to idenBuckets. We put them at
    # opposite sides of the lists to make the identity linking step easier.
    idenBuckets = [[b1]] + idenBuckets + [[b2]]

    # Cut link
    loop2.network.removeLink(b1.link)    

    # Link inner identities
    for i in range(len(idenBuckets)-1):
        idenB1 = idenBuckets[i][0]
        idenB2 = idenBuckets[i+1][0]
        Link(idenB1, idenB2)
        idenBuckets[i+1].remove(idenB2)

    
    # Add idenNodes to network. They don't have any external buckets
    # so the NetworkTensor container doesn't need to know about them.

    for idenNode in idenNodes:
        loop2.network.addNode(idenNode)

    # Merge identity tensors with the loop
    activeNode = n1
    done = set()
    while True:
        connected = activeNode.connectedNodes
        if n2 in connected:
            break
        nextNode = connected.difference(idenNodes).difference(done).pop()
        nextIden = connected.intersection(idenNodes).difference(done).pop()
        done.add(activeNode)
        activeNode = loop2.network.mergeNodes(nextNode, nextIden)

    # Merge links
    for n in loop2.network.nodes:
        loop2.network.mergeLinks(n, compress=False)

    # Contract rank-2 tensors
    loop2.network.contractRank2()

    return loop2

def prepareEnvironment(node1, node2, tensor, externalEnvironment, bids, otherBids):
    '''
    Computes the environment of the specified Nodes in the specified NetworkTensor.
    The NetworkTensor is assumed to have an external environment associated with its
    external Buckets and that is accounted for as well.

    :param node1: The first Node to consider.
    :param node2: The second Node to consider.
    :param tensor: The tensor containing node1 and node2 to use in computing the environment.
    :param externalEnvironment: The external environment of tensor.
    :param bids: Bucket IDs in tensor.
    :param otherBids: Bucket IDs in externalEnvironment. Ordered corresponding to bids.
    :return: environment - A dictionary key'd by bucketID with tensors as values.
                    The tensor in each case must be rank 2 and symmetric, and bucketID
                    is the ID of the Bucket to which the tensor attaches. These Buckets
                    must be on one of node1 or node2. The tensors represent the environment
                    of their respective Buckets, including both other nodes (other than
                    node1 and node2) and the externalEnvironment.
    '''
    
    # Get Node IDs
    nid1 = node1.id
    nid2 = node2.id
    
    # Make list of Bucket IDs on node1 and node2 that do not link them to each other.
    buckets1 = node1.buckets
    buckets2 = node2.buckets

    bids1 = list(b.id for b in buckets1 if not b.linked or b.otherNode != node2)
    bids2 = list(b.id for b in buckets2 if not b.linked or b.otherNode != node1)

    # Contract the tensor against the external environment and then against itself again.
    t2, nidDict, bidDict = tensor.copy()
    
    indsEnv = list(range(tensor.rank))
    for i,b in enumerate(externalEnvironment.externalBuckets):
        if b.id in otherBids:
            indsEnv[otherBids.index(b.id)] = i

    net = tensor.contract(range(tensor.rank), externalEnvironment, indsEnv, elimLoops=False)
    net = net.contract(indsEnv, t2, range(tensor.rank), elimLoops=False)

    # Dictionary for moving from buckets in the environment to those in node1 and node2
    bDict = {}
    for n in net.network.nodes:
        for b in n.buckets:
            if b.id in bids1 or b.id in bids2:
                bDict[b.otherBucket.id] = b.id

    # Remove node1 and node2 and their doubles
    removeIDs = set()
    removeIDs.add(nid1)
    removeIDs.add(nid2)
    removeIDs.add(nidDict[nid1])
    removeIDs.add(nidDict[nid2])
    toRemove = []
    for n in net.network.nodes:
        if n.id in removeIDs:
            toRemove.append(n)
    for n in toRemove:
        net.removeNode(n)

    # Contract net down
    # The environment weighting does not care about the scale of the environment
    # matrices (they all end up multiplying the same objects, so it's an overall
    # scale factor). We therefore ignore these.
    arrs, buckets, _ = net.disjointArrays

    # Should be four disjoint components, corresponding to the environments
    # of the four Buckets we are interested in.
    assert len(arrs) == 4

    # The arrays should be symmetric and rank 2.
    for a in arrs:
        assert len(a.shape) == 2
        if L2error(a, a.T) > config.runParams['epsilon']:
            logger.warning('Environment not symmetric. Violation ' + str(L2error(a, a.T))) 

    
    # Associate arrays with buckets
    
    environment = {}
    for arr, bList in zip(*(arrs, buckets)):
        for b in bList:
            if b in bDict:
                environment[bDict[b]] = arr
            
    return environment

def environmentSVD(node1, node2, environment, accuracy, rank):
    '''
    Compresses the link between node1 and node2, accounting for the environment.
    :param node1: The first Node. Must be linked to node2.
    :param node2: The second Node. Must be linked to node1.
    :param environment: A dictionary of {Bucket ID: environment matrix} pairs.
                    All Bucket ID's must be represented among the Buckets of node1
                    and node2. All Bucket ID's among those of node1 and node2 must
                    be represented other than the ones linking them.
    :param accuracy: The accuracy of the compression.
    :return: Two ArrayTensors, representing the compressed tensors of node1 and node2
            in that order.
    '''
    
    # Figure out which indices link the two Nodes
    ind1, ind2 = node1.indicesConnecting(node2) 
    ind1 = ind1[0]
    ind2 = ind2[0]
    
    # Order environment tensors
    env1 = []
    for i,b in enumerate(node1.buckets):
        if i != ind1:
            env1.append(environment[b.id])

    env2 = []
    for i,b in enumerate(node2.buckets):
        if i != ind2:
            env2.append(environment[b.id])

    assert len(env1) == 2
    assert len(env2) == 2

    # Form environment arrays
    envArr1 = np.einsum('ij,kl->ikjl', env1[0], env1[1])
    envArr2 = np.einsum('ij,kl->ikjl', env2[0], env2[1])

    # Flatten
    envArr1 = np.reshape(envArr1, (envArr1.shape[0]*envArr1.shape[1], envArr1.shape[2]*envArr1.shape[3]))
    envArr2 = np.reshape(envArr2, (envArr2.shape[0]*envArr2.shape[1], envArr2.shape[2]*envArr2.shape[3]))

    # Contract arrays
    arr1 = node1.tensor.array
    arr2 = node2.tensor.array
    
    einArgs1 = [0,1,2]
    einArgs2 = [3,4,5]
    einArgs2[ind2] = einArgs1[ind1]
    net = np.einsum(arr1, einArgs1, arr2, einArgs2)
        
    # Store shapes for un-flattening
    sh1 = net.shape[:2]
    sh2 = net.shape[2:]
    
    # Flatten
    oldNet = np.array(net)
    net = np.reshape(net, (net.shape[0]*net.shape[1], net.shape[2]*net.shape[3]))
    
    # SVD
    envArr1 = np.identity(envArr1.shape[0])
    envArr2 = np.identity(envArr2.shape[0])
#    print(environment)
#    print('env',envArr1)
#    print('env',envArr2)
#    assert np.sum((envArr1 - np.identity(len(envArr1)))**2) < 1e-10
#    assert np.sum((envArr2 - np.identity(len(envArr2)))**2) < 1e-10
    A, B = svd.environmentSVD(net, envArr1, envArr2, accuracy)

    logger.debug('Predicted rank is ' + str(rank) + ', actual ' + str(A.shape[1]))

    # Un-flatten
    A = np.reshape(A, list(sh1) + [A.shape[1]])
    B = np.reshape(B, list(sh2) + [B.shape[1]])

    # Permute indices
    if ind1 == 0:
        A = np.transpose(A, axes=(2,0,1))
    elif ind1 == 1:
        A = np.transpose(A, axes=(0,2,1))
    elif ind1 == 2:
        A = np.transpose(A, axes=(0,1,2))

    if ind2 == 0:
        B = np.transpose(B, axes=(2,0,1))
    elif ind2 == 1:
        B = np.transpose(B, axes=(0,2,1))
    elif ind2 == 2:
        B = np.transpose(B, axes=(0,1,2))

    return A, B
    
def svdCut(loop, environment, link, bids, otherBids, rankDict):
    lbids = list(b.id for b in loop.externalBuckets)
    ebids = list(b.id for b in environment.externalBuckets)
    for i in range(len(bids)):
        ind1 = lbids.index(bids[i])
        ind2 = ebids.index(otherBids[i])
        assert loop.externalBuckets[ind1].size == environment.externalBuckets[ind2].size
    
    # Perform trivial cut
    loop2 = trivialCut(loop, link)
    
    # Gather links
    links = set()
    for n in loop2.network.nodes:
        for b in n.buckets:
            if b.linked:
                links.add(b.link)

    # Contract the environment against itself
    inds = []
    for i in range(environment.rank):
        if environment.externalBuckets[i].id not in otherBids:
            inds.append(i)

    newEnv, _, _ = environment.copy()
    environment = environment.contract(inds, newEnv, inds, elimLoops=False)
    environment.contractRank2()

    ebids = list(b.id for b in environment.externalBuckets)
    for i in range(len(bids)):
        ind1 = lbids.index(bids[i])
        ind2 = ebids.index(otherBids[i])
        assert loop.externalBuckets[ind1].size == environment.externalBuckets[ind2].size

    # Optimize
    for l in links:
        node1 = l.bucket1.node
        node2 = l.bucket2.node

        freeBuckets = lambda x: list(b for b in x.buckets if not b.linked)

        leftBids = set()
        current = node1
        prev = node2
        done = False
        while not done:
            fb = freeBuckets(current)
            for f in fb:
                leftBids.add(f.id)
            found = False
            for n in current.connectedNodes:
                if n != prev:
                    prev = current
                    current = n
                    found = True
                    break
            
            if not found:
                done = True        
        
        rightBids = set(b.id for b in loop2.externalBuckets).difference(leftBids)

        leftBids = frozenset(leftBids)
        rightBids = frozenset(rightBids)
        
        rank = rankDict[(leftBids, rightBids)]
            
        # Calculate environment tensors
        env = prepareEnvironment(node1, node2, loop2, environment, bids, otherBids)

        # The rank corrects for the fact that we will incur this error multiple times as 
        # we go through the loop truncating. For a discussion see doi:10.1137/090752286.
        A, B = environmentSVD(node1, node2, env, loop.accuracy / loop.rank, rank)
        
        print(A.shape, B.shape)
        node1.tensor = ArrayTensor(A)
        node2.tensor = ArrayTensor(B)
        
    
    return loop2
