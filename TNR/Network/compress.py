import numpy as np
from copy import deepcopy

from TNR.Utilities.arrays import ndArrayToMatrix, matrixToNDArray
from TNR.Utilities.svd import svdByPrecision, matrixProductLinearOperator

def compress_link_in_network(network, l, accuracy, return_copy):
    if return_copy:
        network = deepcopy(network)
        n1 = l.bucket1.node
        n2 = l.bucket2.node
        node1 = list(n for n in network.nodes if n.id == n1.id)[0]
        node2 = list(n for n in network.nodes if n.id == n2.id)[0]
        l = node1.linksConnecting(node2)

    compressLink(l, accuracy)

    return network

def compressLink(l, accuracy):
    b1 = l.bucket1
    b2 = l.bucket2

    ind1 = b1.index
    ind2 = b2.index

    n1 = b1.node
    n2 = b2.node

    arr1, ind1I = n1.tensor.getIndexFactor(ind1)
    arr2, ind2I = n2.tensor.getIndexFactor(ind2)

    sh1 = list(arr1.shape)
    sh2 = list(arr2.shape)

    sh1m = sh1[:ind1I] + sh1[ind1I + 1:]
    sh2m = sh2[:ind2I] + sh2[ind2I + 1:]

    a1 = ndArrayToMatrix(arr1, ind1I, front=False)
    a2 = ndArrayToMatrix(arr2, ind2I, front=True)

    if a1.shape[1] < a1.shape[0] and a2.shape[0] < a2.shape[1]:
        arr = matrixProductLinearOperator(a1, a2)
        u, lam, v = svdByPrecision(arr, accuracy, True)
    else:
        arr = np.dot(a1, a2)
        u, lam, v = svdByPrecision(arr, accuracy, True)

    p = lam**2
    p /= np.sum(p)
    cp = np.cumsum(p)

    ind = np.searchsorted(cp, accuracy, side='left')
    ind = len(cp) - ind

    u = u[:, :ind]
    lam = lam[:ind]
    v = v[:ind, :]

    u *= np.sqrt(lam)[np.newaxis, :]
    v *= np.sqrt(lam)[:, np.newaxis]

    uu = matrixToNDArray(u, sh1[:ind1I] + [ind] +
                         sh1[ind1I + 1:], ind1I, front=False)
    vv = matrixToNDArray(v, sh2[:ind2I] + [ind] +
                         sh2[ind2I + 1:], ind2I, front=True)

    print('Compress:', l.bucket1.size, accuracy, ind, cp)

    n1.tensor = n1.tensor.setIndexFactor(ind1, uu)
    n2.tensor = n2.tensor.setIndexFactor(ind2, vv)

    assert n1.tensor.shape[ind1] == n2.tensor.shape[ind2]
