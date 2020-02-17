import numpy as np
from collections import defaultdict

from TNR.Network.link import Link
from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Tensor.arrayTensor import ArrayTensor


def layer(n):
    return int(np.log2(n / 3)) + 2

def IdentityTensor(dimension, rank, accuracy):
    '''
    Builds a TreeTensor representing the identity.
    :param dimension: The dimension of the bonds in the tree.
    :param rank: The rank of the tree.
    :param accuracy: The accuracy of the tree.
    :return: Identity tensor.
    '''
    
    tens = TreeTensor(accuracy)
    
    if rank == 0:
        tens.addTensor(ArrayTensor(np.array(1.)))
    if rank == 1:
        tens.addTensor(ArrayTensor(np.ones(dimension)))
    elif rank == 2:
        tens.addTensor(ArrayTensor(np.identity(dimension)))
    else:
        numTensors = rank - 2

        buckets = []

        # Create identity array
        iden = np.zeros((dimension, dimension, dimension))
        for i in range(dimension):
            iden[i, i, i] = 1.0

        for i in range(numTensors):
            n = tens.addTensor(ArrayTensor(iden))
            buckets = buckets + n.buckets

        while len(tens.network.externalBuckets) > rank:
            b = buckets.pop(0)
            i = 0
            while buckets[i].node is b.node or len(
                    buckets[i].node.connectedNodes) > 0:
                i += 1
            Link(b, buckets[i])

            tens.externalBuckets.remove(b)
            tens.externalBuckets.remove(buckets[i])
            tens.network.externalBuckets.remove(b)
            tens.network.externalBuckets.remove(buckets[i])
            tens.network.internalBuckets.add(b)
            tens.network.internalBuckets.add(buckets[i])

            buckets.remove(buckets[i])
        
    return tens