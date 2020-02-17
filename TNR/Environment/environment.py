import numpy as np
import networkx

from copy import deepcopy

from TNR.Network.network import Network
from TNR.Network.node import Node
from TNR.NetworkTensor.networkTensor import NetworkTensor
from TNR.Tensor.arrayTensor import ArrayTensor



def artificialCut(tensor, body):
    '''
    Produces an estimate of the environment of body within tensor.
    This is done by cutting a random minimal set of bonds in the network excluding body,
    turning the associated buckets into external ones rather than internal ones.

    By convention if a node within body has an external bucket the environment
    will contain an identity tensor which connects to it.

    :param body: The nodes the environment of which to compute.
    :param tensor: The tensor containing the nodes in body.

    :return environment: The TreeTensor containing an estimate of the environment of body.
    :return bodyTensor: The NetworkTensor containing the body. This is derived from a deepcopy of tensor.
    :return internalBids: The ID's of the internal buckets in bodyTensor.
    :return envBids: A list of ID's of the external buckets of environment which correspond in order to the
                    external buckets of bodyTensor.
    '''

    # Ensures we don't break the original
    environment = deepcopy(tensor)

    # Helper lists
    nodes2 = list(environment.network.nodes)
    excludeIDs = list(n.id for n in body)
            
    # Prepare graph
    indicator = lambda x: -1e100 if x[0].id in excludeIDs and x[1].id in excludeIDs else 0
    g = networkx.Graph()
    g.add_nodes_from(environment.network.nodes)
    for n in environment.network.nodes:
        for m in environment.network.internalConnected(n):
            g.add_edge(n, m, weight=indicator((n,m)))

    # Make spanning tree
    tree = networkx.minimum_spanning_tree(g)
    
    # Cut all links not in the spanning tree
    internalBids = []
    for n in nodes2:
        for b in n.buckets:
            if b.linked:
                n2 = b.otherNode
                if (n,n2) not in tree.edges:
                    environment.externalBuckets.append(b.link.bucket1)
                    environment.externalBuckets.append(b.link.bucket2)
                    if n2.id in excludeIDs and n.id not in excludeIDs:
                        internalBids.append(b.otherBucket.id)
                    elif n.id in excludeIDs and n2.id not in excludeIDs:
                        internalBids.append(b.id)
                    elif n.id in excludeIDs and n2.id in excludeIDs:
                        assert len(n.findLinks(n2)) == 1
                    environment.network.removeLink(b.link)

    # Remove the body
    for n in nodes2:
        if n.id in excludeIDs:
            environment.removeNode(n)

    # Remove outgoing legs
    for b in environment.externalBuckets:
        b.link = None

    # Create body tensor
    bodyTensor = tensor.copySubset(body)

    # Associate bucket indices between the body and the environment
    envBids = []
    for i,b in enumerate(bodyTensor.externalBuckets):
        ind = list(l.id for l in body).index(b.node.id)
        ind2 = list(b2.id for b2 in body[ind].buckets).index(b.id)
        if body[ind].buckets[ind2].linked and b.id not in internalBids:
            envBids.append(body[ind].buckets[ind2].otherBucket.id)
        else:
            # Append identity's to go with buckets in body that are external to tensor.
            n = Node(ArrayTensor(np.identity(bodyTensor.externalBuckets[i].size)))
            environment.network.addNode(n)
            environment.externalBuckets.append(n.buckets[0])
            environment.externalBuckets.append(n.buckets[1])
            # We've just added two buckets, so we associate one with the body
            # and one with the environment
            envBids.append(n.buckets[0].id)

    return environment, bodyTensor, internalBids, envBids


def fullEnvironment(tensor, body):
    '''
    Produces the environment of body within tensor.

    By convention if a node within body has an external bucket the environment
    will contain an identity tensor which connects to it.

    :param body: The nodes the environment of which to compute.
    :param tensor: The tensor containing the nodes in body.

    :return environment: The TreeTensor containing an estimate of the environment of body.
    :return bodyTensor: The NetworkTensor containing the body. This is derived from a deepcopy of tensor.
    :return internalBids: The ID's of the internal buckets in bodyTensor.
    :return envBids: A list of ID's of the external buckets of environment which correspond in order to the
                    external buckets of bodyTensor.
    '''

    # Ensures we don't break the original
    environment = deepcopy(tensor)

    # Helper lists
    nodes2 = list(environment.network.nodes)
    excludeIDs = list(n.id for n in body)

    # Cut all links not in the spanning tree
    internalBids = []
    for n in nodes2:
        for b in n.buckets:
            if b.linked:
                n2 = b.otherNode
                if n2.id in excludeIDs and n.id not in excludeIDs:
                    internalBids.append(b.otherBucket.id)
                elif n.id in excludeIDs and n2.id not in excludeIDs:
                    internalBids.append(b.id)
                elif n.id in excludeIDs and n2.id in excludeIDs:
                    assert len(n.findLinks(n2)) == 1

    # Remove the body
    for n in nodes2:
        if n.id in excludeIDs:
            environment.removeNode(n)

    # Remove outgoing legs
    for b in environment.externalBuckets:
        b.link = None

    # Create body tensor
    bodyTensor = tensor.copySubset(body)

    # Associate bucket indices between the body and the environment
    envBids = []
    for i,b in enumerate(bodyTensor.externalBuckets):
        ind = list(l.id for l in body).index(b.node.id)
        ind2 = list(b2.id for b2 in body[ind].buckets).index(b.id)
        if body[ind].buckets[ind2].linked and b.id not in internalBids:
            envBids.append(body[ind].buckets[ind2].otherBucket.id)
        else:
            # Append identity's to go with buckets in body that are external to tensor.
            n = Node(ArrayTensor(np.identity(bodyTensor.externalBuckets[i].size)))
            environment.network.addNode(n)
            environment.externalBuckets.append(n.buckets[0])
            environment.externalBuckets.append(n.buckets[1])
            # We've just added two buckets, so we associate one with the body
            # and one with the environment
            envBids.append(n.buckets[0].id)

    return environment, bodyTensor, internalBids, envBids

def identityEnvironment(tensor, body):
    '''
    Produces an artificial environment tensor containing one appropriately shaped identity matrix
    on each external leg of the specified body.

    :param body: The nodes the environment of which to compute.
    :param tensor: The tensor containing the nodes in body.

    :return environment: The TreeTensor containing an estimate of the environment of body.
    :return bodyTensor: The NetworkTensor containing the body. This is derived from a deepcopy of tensor.
    :return internalBids: The ID's of the internal buckets in bodyTensor.
    :return envBids: A list of ID's of the external buckets of environment which correspond in order to the
                    external buckets of bodyTensor. 
    '''

    bodyTensor = tensor.copySubset(body)

    environment = NetworkTensor(tensor.accuracy)

    internalBids = []
    for n in bodyTensor.network.nodes:
        for b in n.buckets:
            if b not in bodyTensor.externalBuckets:
                internalBids.append(b.id)

    envBids = []
    for b in bodyTensor.externalBuckets:
        n = environment.addTensor(ArrayTensor(np.identity(b.size)))
        envBids.append(n.buckets[0].id)

    return environment, bodyTensor, internalBids, envBids

