import numpy as np
from scipy.linalg import sqrtm
from copy import deepcopy
from TNR.Network.link import Link
from TNR.Network.node import Node
from TNR.Tensor.arrayTensor import ArrayTensor

def trivialCut(loop, link):
    '''
    Cuts the loop by running the specified index through a series of
    identity tensors.
    :param loop: NetworkTensor specifying a loop.
    :param link: The link to cut. 
    :return: A cut copy of the NetworkTensor.
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
    while True:
        connected = activeNode.connectedNodes
        if n2 in connected:
            break
        nextNode = connected.difference(idenNodes).pop()
        nextIden = connected.intersection(idenNodes).pop()
        activeNode = loop2.network.mergeNodes(nextNode, nextIden)    
    
    print(loop2)
    
    # Merge links
    for n in loop2.network.nodes:
        loop2.network.mergeLinks(n, compress=False)
    
    return loop2

def cutSVD(loop, environment, ranks):
    '''
    Uses SVD to cut the loop along the bond between loop[0] and loop[-1].
    
    :param loop: A list of arrays of rank 3. These are ordered such that the
                last dimension of each contracts with the first of the next.
    :param environment: A list of arrays of rank 2. These are symmetric and ordered
                such that the middle dimension on loop[i] contracts with one index of
                environment[i].
    :param ranks: A list of post-cut ranks the SVD will use. These are ordered so that
                ranks[i] is the dimension of the bond between loop[i] and loop[i+1].
    :return: List of arrays, the first and last of which are rank 2. The remainder are
                rank 3. These are ordered corresponding to those in loop and represent
                the same overall tensor.
    '''
    
    # Size of cut bond
    cutSize = loop[0].shape[0]
    
    # Tensor product with the identity.
    idenLoop = list(np.einsum('ijk,lm->lijkm', loop[i], np.identity(cutSize)) for i in range(len(loop)))
    
    # Contract bond-to-cut on ends
    idenLoop[0] = np.einsum('iijkm->jkm',idenLoop[0])
    idenLoop[-1] = np.eisnum('lijkk->lij', idenLoop[-1])
    
    # Merge bonds on ends
    idenLoop[0] = np.reshape(idenLoop[0], (idenLoop[0].shape[0], idenLoop[0].shape[1]*idenLoop[0].shape[2]))
    idenLoop[-1] = np.reshape(idenLoop[-1], (idenLoop[-1].shape[0]*idenLoop[-1].shape[1], idenLoop[-1].shape[2]))

    # Merge internal bonds
    for i in range(1, len(idenLoop) - 1):
        idenLoop[i] = np.reshape(idenLoop[i], (idenLoop[i].shape[0]*idenLoop[i].shape[1], idenLoop[i].shape[2], idenLoop[i].shape[3]*idenLoop[i].shape[4]))
    
    # SVD left end
    
    envLeft = sqrtm(environment[0])
    envRight = sqrtm(environment[1])
    