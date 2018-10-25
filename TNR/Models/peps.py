import numpy as np
from scipy.integrate import quad

from TNR.Network.link import Link
from TNR.Network.node import Node
from TNR.Network.network import Network
from TNR.NetworkTensor.networkTensor import NetworkTensor
from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Tensor.arrayTensor import ArrayTensor


def wrapper(i, n):
    if i < 0:
        return i + n
    elif i >= n:
        return i - n
    else:
        return i

def treeify(x, accuracy):
    t = ArrayTensor(x)
    tt = TreeTensor(accuracy)
    tt.addTensor(t)
    return tt

def featureless_bosonic_insulator():
    Iden = np.eye(2)
    Flip = np.array([0, 1, 1, 0]).reshape(2, 2)
    Alpha = np.array([1, 0, 0, 1, 0, 0, 1, 0]).reshape(2, 2, 2)
    Delta = np.zeros([4, 2, 2, 2])
    sqrt_factorial = {0: 1, 1: 1, 2: np.sqrt(2), 3: np.sqrt(6)}
    for p in range(4):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if p == i + j + k:
                        Delta[p, i, j, k] = sqrt_factorial[p]

    A = np.reshape(np.einsum("ixZ,jyX,kz,pijk->pxXyzZ", Alpha, Alpha,
                             Iden, Delta),
                   (4, 4, 2, 4))
    B = np.reshape(np.einsum("ix,jZy,kXz,pijk->pxXyzZ", Flip, Alpha,
                             Alpha, Delta), (4, 4, 2, 4))
    return A, B


def aklt2d():
    D = np.zeros([4, 2, 2, 2])
    for p in range(4):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if p == i + j + k:
                        D[p, i, j, k] = 1

    coeffA = np.zeros([4, 4])
    coeffA[0, 0] = -1
    coeffA[1, 1] = np.sqrt(1 / 3)
    coeffA[2, 2] = -np.sqrt(1 / 3)
    coeffA[3, 3] = 1
    coeffB = np.zeros([4, 4])
    coeffB[3, 0] = 1
    coeffB[2, 1] = np.sqrt(1 / 3)
    coeffB[1, 2] = np.sqrt(1 / 3)
    coeffB[0, 3] = 1

    A = np.einsum("pq,qxyz->pxyz", coeffA, D)
    B = np.einsum("pq,qxyz->pxyz", coeffB, D)
    return A, B


def featureless_bosonic_insulator_hardcore():
    Iden = np.eye(2)
    Flip = np.array([0, 1, 1, 0]).reshape(2, 2)
    Alpha = np.array([1, 0, 0, 1, 0, 0, 1, 0]).reshape(2, 2, 2)
    Delta = np.zeros([2, 2, 2, 2])
    for p in range(2):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if p == i + j + k:
                        Delta[p, i, j, k] = 1

    A = np.reshape(np.einsum("ixZ,jyX,kz,pijk->pxXyzZ", Alpha, Alpha,
                             Iden, Delta),
                   (2, 4, 2, 4))
    B = np.reshape(np.einsum("ix,jZy,kXz,pijk->pxXyzZ", Flip, Alpha,
                             Alpha, Delta), (2, 4, 2, 4))
    return A, B


def featureless_su2(c1, c2, c3, c4):
    singlet = np.array([[0, 1], [-1, 0]])  # = |01> - |10>
    assert singlet[0, 1] == 1
    assert singlet[1, 0] == -1

    singlet_11 = np.zeros((4, 4))
    singlet_11[:2, :2] = singlet
    singlet_12 = np.zeros((4, 4))
    singlet_12[:2, 2:] = singlet
    singlet_21 = np.zeros((4, 4))
    singlet_21[2:, :2] = singlet
    singlet_22 = np.zeros((4, 4))
    singlet_22[2:, 2:] = singlet

    singlet__1 = np.zeros((2, 4))
    singlet__1[:, :2] = singlet

    singlet__2 = np.zeros((2, 4))
    singlet__2[:, 2:] = singlet

    b1 = np.einsum("ab,cd->abcd", singlet__2, singlet_11)
    b2 = np.einsum("ab,cd->abcd", singlet__1, singlet_22)
    b3p = np.einsum("ab,cd->abcd", singlet__1, singlet_12)
    b3pp = np.einsum("ab,cd->abcd", singlet__1, singlet_21)
    b3 = b3p - b3pp
    b4p = np.einsum("ab,cd->abcd", singlet__2, singlet_21)
    b4pp = np.einsum("ab,cd->abcd", singlet__2, singlet_12)
    b4 = b4p - b4pp
    b5 = np.einsum("ab,cd->abcd", singlet__1, singlet_11)

    B1 = b1 + np.transpose(b1, (0, 2, 3, 1)) + np.transpose(b1, (0,
                                                                 3, 1, 2))
    B2 = b2 + np.transpose(b2, (0, 2, 3, 1)) + np.transpose(b2, (0,
                                                                 3, 1, 2))
    B3 = b3 + np.transpose(b3, (0, 2, 3, 1)) + np.transpose(b3, (0,
                                                                 3, 1, 2))
    B4 = b4 + np.transpose(b4, (0, 2, 3, 1)) + np.transpose(b4, (0,
                                                                 3, 1, 2))

    B5 = b5 + np.transpose(b5, (0, 2, 3, 1)) + np.transpose(b5, (0,
                                                                 3, 1, 2))
    assert np.allclose(B5, np.zeros(B5.shape))

    assert np.allclose(B1 + np.transpose(B1, (0, 2, 1, 3)),
                       np.zeros(B1.shape))
    assert np.allclose(B2 + np.transpose(B2, (0, 2, 1, 3)),
                       np.zeros(B2.shape))
    assert np.allclose(B3 - np.transpose(B3, (0, 2, 1, 3)),
                       np.zeros(B3.shape))
    assert np.allclose(B4 - np.transpose(B4, (0, 2, 1, 3)),
                       np.zeros(B4.shape))

    bond_singlet = singlet_11 + singlet_22

    # print "c1={}, c2={}, c3={}, c4={}".format(c1, c2, c3, c4)
    SA = c1 * B1 + c2 * B2 + c3 * B3 + c4 * B4
    SB = np.einsum("pxyz,xX,yY,zZ->pxyz", SA, bond_singlet,
                   bond_singlet, bond_singlet)
    return (SA, SB)


def peps2Dhoneycomb(nX, nY, A, B, accuracy):
    '''

    :param nX: Size along one dimension.
    :param nY: Size along the other dimension.
    :param A: Rank-4 array specifying the A-sites. The first index is physical.
    :param B: Rank-4 array specifying the B-sites. The first index is physical.
    :param accuracy: 
    :return: Network encoding the specified peps.
    '''

    network = Network()

    # Place to store the tensors
    latticeA = [[] for i in range(nX)]
    latticeB = [[] for i in range(nY)]
    latticeAp = [[] for i in range(nX)]
    latticeBp = [[] for i in range(nY)]

    # Add tensors
    for i in range(nX):
        for j in range(nY):
            latticeA[i].append(Node(treeify(A, accuracy)))
            latticeB[i].append(Node(treeify(B, accuracy)))
            latticeAp[i].append(Node(treeify(A, accuracy)))
            latticeBp[i].append(Node(treeify(B, accuracy)))

    # Generate links
    for i in range(nX):
        for j in range(nY):
            # A only links to B and vice-versa.

            # Conveniently the indices are arranged so that they correspond
            # directly (i.e. 1->1, 2->2, 3->3).

            # We take the convention that the X link between A and B occurs at
            # the same i and j, and that the Y link from A to B occurs at the same j
            # but i+1. Hence the Z link from A to B occurs at i+1,j-1.

            # X
            b1 = latticeA[i][j].buckets[1]
            b2 = latticeB[i][j].buckets[1]
            Link(b1, b2)

            b1 = latticeAp[i][j].buckets[1]
            b2 = latticeBp[i][j].buckets[1]
            Link(b1, b2)

            # Y
            b1 = latticeA[i][j].buckets[2]
            b2 = latticeB[(i + 1) % nX][j].buckets[2]
            Link(b1, b2)

            b1 = latticeAp[i][j].buckets[2]
            b2 = latticeBp[(i + 1) % nX][j].buckets[2]
            Link(b1, b2)

            # Z
            b1 = latticeA[i][j].buckets[3]
            b2 = latticeB[(i + 1) % nX][wrapper(j - 1, nY)].buckets[3]
            Link(b1, b2)

            b1 = latticeAp[i][j].buckets[3]
            b2 = latticeBp[(i + 1) % nX][wrapper(j - 1, nY)].buckets[3]
            Link(b1, b2)

            # Physical
            b1 = latticeA[i][j].buckets[0]
            b2 = latticeAp[i][j].buckets[0]
            Link(b1, b2)

            b1 = latticeB[i][j].buckets[0]
            b2 = latticeBp[i][j].buckets[0]
            Link(b1, b2)

    # Add to network
    for i in range(nX):
        for j in range(nY):
            network.addNode(latticeA[i][j])
            network.addNode(latticeB[i][j])
            network.addNode(latticeAp[i][j])
            network.addNode(latticeBp[i][j])

    return network


def single_honeycomb(nX, nY, A, B, accuracy):
    '''

    :param nX: Size along one dimension.
    :param nY: Size along the other dimension.
    :param A: Rank-4 array specifying the A-sites. The first index is physical.
    :param B: Rank-4 array specifying the B-sites. The first index is physical.
    :param accuracy: 
    :return: Network encoding the specified peps.
    '''

    network = Network()

    # Place to store the tensors
    latticeA = [[] for i in range(nX)]
    latticeB = [[] for i in range(nY)]

    # Add tensors
    for i in range(nX):
        for j in range(nY):
            latticeA[i].append(Node(treeify(A, accuracy)))
            latticeB[i].append(Node(treeify(B, accuracy)))

    # Generate links
    for i in range(nX-1):
        for j in range(nY-1):
            # A only links to B and vice-versa.

            # Conveniently the indices are arranged so that they correspond
            # directly (i.e. 1->1, 2->2, 3->3).

            # We take the convention that the X link between A and B occurs at
            # the same i and j, and that the Y link from A to B occurs at the same j
            # but i+1. Hence the Z link from A to B occurs at i+1,j-1.

            # X
            b1 = latticeA[i][j].buckets[1]
            b2 = latticeB[i][j].buckets[1]
            Link(b1, b2)


            # Y
            b1 = latticeA[i][j].buckets[2]
            b2 = latticeB[(i + 1) % nX][j].buckets[2]
            Link(b1, b2)

            # Z
            b1 = latticeA[i][j].buckets[3]
            b2 = latticeB[(i + 1) % nX][wrapper(j - 1, nY)].buckets[3]
            Link(b1, b2)


    # Add to network
    for i in range(nX):
        for j in range(nY):
            network.addNode(latticeA[i][j])
            network.addNode(latticeB[i][j])

    return network

