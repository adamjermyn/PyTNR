import numpy as np
from scipy.integrate import quad

from TNR.Network.link import Link
from TNR.Network.node import Node
from TNR.Network.network import Network
from TNR.TreeTensor.identityTensor import IdentityTensor
from TNR.Tensor.arrayTensor import ArrayTensor


def IsingModel1D(nX, h, J, accuracy):
    network = Network()

    # Place to store the tensors
    lattice = []
    onSite = []
    bond = []

    # Each lattice site has seven indices of width five, and returns zero if
    # they are unequal and one otherwise.
    for i in range(nX):
        lattice.append(Node(IdentityTensor(2, 3, accuracy=accuracy)))

    # Each on-site term has one index of width two, and returns exp(-h) or
    # exp(h) for 0 or 1 respectively.
    for i in range(nX):
        arr = np.zeros((2))
        arr[0] = np.exp(-h)
        arr[1] = np.exp(h)
        onSite.append(Node(ArrayTensor(arr)))

    # Each bond term has two indices of width two and returns
    # exp(-J*(1+delta(index0,index1))/2).
    for i in range(nX):
        arr = np.zeros((2, 2))
        arr[0][0] = np.exp(-J)
        arr[1][1] = np.exp(-J)
        arr[0][1] = np.exp(J)
        arr[1][0] = np.exp(J)
        bond.append(Node(ArrayTensor(arr)))

    # Attach links
    for i in range(nX):
        Link(lattice[i].buckets[0], onSite[i].buckets[0])
        Link(lattice[i].buckets[1], bond[i].buckets[0])
        Link(lattice[i].buckets[2], bond[(i + 1) % nX].buckets[1])

    # Add to Network
    for i in range(nX):
        network.addNode(lattice[i])
        network.addNode(onSite[i])
        network.addNode(bond[i])

    return network


def IsingModel1Ddisordered(nX, h0, J0, accuracy):
    network = Network()

    # Place to store the tensors
    lattice = []
    onSite = []
    bond = []

    # Each lattice site has seven indices of width five, and returns zero if
    # they are unequal and one otherwise.
    for i in range(nX):
        lattice.append(Node(IdentityTensor(2, 3, accuracy=accuracy)))

    # Each on-site term has one index of width two, and returns exp(-h) or
    # exp(h) for 0 or 1 respectively.
    for i in range(nX):
        h = h0 * np.random.randn()
        arr = np.zeros((2))
        arr[0] = np.exp(-h)
        arr[1] = np.exp(h)
        onSite.append(Node(ArrayTensor(arr)))

    # Each bond term has two indices of width two and returns
    # exp(-J*(1+delta(index0,index1))/2).
    for i in range(nX):
        J = J0 * np.random.randn()
        arr = np.zeros((2, 2))
        arr[0][0] = np.exp(-J)
        arr[1][1] = np.exp(-J)
        arr[0][1] = np.exp(J)
        arr[1][0] = np.exp(J)
        bond.append(Node(ArrayTensor(arr)))

    # Attach links
    for i in range(nX):
        Link(lattice[i].buckets[0], onSite[i].buckets[0])
        Link(lattice[i].buckets[1], bond[i].buckets[0])
        Link(lattice[i].buckets[2], bond[(i + 1) % nX].buckets[1])

    # Add to Network
    for i in range(nX):
        network.addNode(lattice[i])
        network.addNode(onSite[i])
        network.addNode(bond[i])

    return network


def exactIsing1Dh(h):
    return np.log(2 * np.cosh(h))


def exactIsing1DJ(n, J):
    J = -J
    l1 = 2 * np.cosh(J)
    l2 = 2 * np.sinh(J)

    q = l2 / l1

    f = 0
    if abs(q)**n < 1e-10:
        f = np.log1p(q**n) / n
    else:
        f = np.log1p((l2 / l1)**n) / n

    return np.log(l1) + f


def IsingModel2D(nX, nY, h, J, accuracy):
    network = Network()

    # Place to store the tensors
    lattice = [[] for i in range(nX)]
    onSite = [[] for i in range(nX)]
    bondV = [[] for i in range(nX)]
    bondH = [[] for i in range(nX)]

    # Each lattice site has seven indices of width five, and returns zero if
    # they are unequal and one otherwise.
    for i in range(nX):
        for j in range(nY):
            lattice[i].append(Node(IdentityTensor(2, 5, accuracy=accuracy)))

    # Each on-site term has one index of width two, and returns exp(-h) or
    # exp(h) for 0 or 1 respectively.
    for i in range(nX):
        for j in range(nY):
            arr = np.zeros((2))
            arr[0] = np.exp(-h)
            arr[1] = np.exp(h)
            onSite[i].append(Node(ArrayTensor(arr)))

    # Each bond term has two indices of width two and returns
    # exp(-J*(1+delta(index0,index1))/2).
    for i in range(nX):
        for j in range(nY):
            arr = np.zeros((2, 2))
            arr[0][0] = np.exp(-J)
            arr[1][1] = np.exp(-J)
            arr[0][1] = np.exp(J)
            arr[1][0] = np.exp(J)
            bondV[i].append(Node(ArrayTensor(arr)))
            bondH[i].append(Node(ArrayTensor(arr)))

    # Attach links
    for i in range(nX):
        for j in range(nY):
            Link(lattice[i][j].buckets[0], onSite[i][j].buckets[0])
            Link(lattice[i][j].buckets[1], bondV[i][j].buckets[0])
            Link(lattice[i][j].buckets[2], bondV[i][(j + 1) % nY].buckets[1])
            Link(lattice[i][j].buckets[3], bondH[i][j].buckets[0])
            Link(lattice[i][j].buckets[4], bondH[(i + 1) % nX][j].buckets[1])

    # Add to Network
    for i in range(nX):
        for j in range(nY):
            network.addNode(lattice[i][j])
            network.addNode(onSite[i][j])
            network.addNode(bondV[i][j])
            network.addNode(bondH[i][j])

    return network


def IsingModel2Dopen(nX, nY, h, J, accuracy):
    network = Network()

    # Place to store the tensors
    lattice = [[] for i in range(nX)]
    onSite = [[] for i in range(nX)]
    bondV = [[] for i in range(nX)]
    bondH = [[] for i in range(nX - 1)]

    counters = [[0 for j in range(nY)] for i in range(nX)]

    # Each lattice site has seven indices of width five, and returns zero if
    # they are unequal and one otherwise.
    lattice[0].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))
    lattice[-1].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))
    for j in range(1, nY - 1):
        lattice[0].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))
        lattice[-1].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))
    lattice[0].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))
    lattice[-1].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))

    for i in range(1, nX - 1):
        lattice[i].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))
        for j in range(1, nY - 1):
            lattice[i].append(Node(IdentityTensor(2, 5, accuracy=accuracy)))
        lattice[i].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))

    # Each on-site term has one index of width two, and returns exp(-h) or
    # exp(h) for 0 or 1 respectively.
    for i in range(nX):
        for j in range(nY):
            arr = np.zeros((2))
            arr[0] = np.exp(-h)
            arr[1] = np.exp(h)
            onSite[i].append(Node(ArrayTensor(arr)))

    # Each bond term has two indices of width two and returns
    # exp(-J*(1+delta(index0,index1))/2).
    for i in range(nX):
        for j in range(nY):
            arr = np.zeros((2, 2))
            arr[0][0] = np.exp(-J)
            arr[1][1] = np.exp(-J)
            arr[0][1] = np.exp(J)
            arr[1][0] = np.exp(J)
            if j < nY - 1:
                bondV[i].append(Node(ArrayTensor(arr)))

            if i < nX - 1:
                bondH[i].append(Node(ArrayTensor(arr)))

    # Attach links
    for i in range(nX):
        for j in range(nY):
            counters[i][j] += 1
            Link(lattice[i][j].buckets[0], onSite[i][j].buckets[0])

    for i in range(nX):
        for j in range(nY):
            print(len(bondH), len(bondV), len(bondV[0]), len(bondV[1]))
            if j > 0:
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondV[i][j - 1].buckets[0])
                counters[i][j] += 1
            if j < nY - 1:
                print(counters[i][j])
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondV[i][j].buckets[1])
                counters[i][j] += 1
            if i > 0:
                print(counters[i][j])
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondH[i - 1][j].buckets[0])
                counters[i][j] += 1
            if i < nX - 1:
                print(counters[i][j])
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondH[i][j].buckets[1])
                counters[i][j] += 1

    # Add to Network
    for x in lattice:
        for y in x:
            network.addNode(y)
    for x in onSite:
        for y in x:
            network.addNode(y)
    for x in bondV:
        for y in x:
            network.addNode(y)
    for x in bondH:
        for y in x:
            network.addNode(y)

    return network


def IsingModel2Ddisordered(nX, nY, h0, J0, accuracy):
    network = Network()

    # Place to store the tensors
    lattice = [[] for i in range(nX)]
    onSite = [[] for i in range(nX)]
    bondV = [[] for i in range(nX)]
    bondH = [[] for i in range(nX - 1)]

    counters = [[0 for j in range(nY)] for i in range(nX)]

    # Each lattice site has seven indices of width five, and returns zero if
    # they are unequal and one otherwise.
    lattice[0].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))
    lattice[-1].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))
    for j in range(1, nY - 1):
        lattice[0].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))
        lattice[-1].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))
    lattice[0].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))
    lattice[-1].append(Node(IdentityTensor(2, 3, accuracy=accuracy)))

    for i in range(1, nX - 1):
        lattice[i].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))
        for j in range(1, nY - 1):
            lattice[i].append(Node(IdentityTensor(2, 5, accuracy=accuracy)))
        lattice[i].append(Node(IdentityTensor(2, 4, accuracy=accuracy)))

    # Each on-site term has one index of width two, and returns exp(-h) or
    # exp(h) for 0 or 1 respectively.
    for i in range(nX):
        for j in range(nY):
            h = h0 * np.random.randn()
            arr = np.zeros((2))
            arr[0] = np.exp(-h)
            arr[1] = np.exp(h)
            onSite[i].append(Node(ArrayTensor(arr)))

    # Each bond term has two indices of width two and returns
    # exp(-J*(1+delta(index0,index1))/2).
    for i in range(nX):
        for j in range(nY):
            J = J0 * np.random.randn()
            arr = np.zeros((2, 2))
            arr[0][0] = np.exp(-J)
            arr[1][1] = np.exp(-J)
            arr[0][1] = np.exp(J)
            arr[1][0] = np.exp(J)
            if j < nY - 1:
                bondV[i].append(Node(ArrayTensor(arr)))

            if i < nX - 1:
                bondH[i].append(Node(ArrayTensor(arr)))

    # Attach links
    for i in range(nX):
        for j in range(nY):
            counters[i][j] += 1
            Link(lattice[i][j].buckets[0], onSite[i][j].buckets[0])

    for i in range(nX):
        for j in range(nY):
            print(len(bondH), len(bondV), len(bondV[0]), len(bondV[1]))
            if j > 0:
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondV[i][j - 1].buckets[0])
                counters[i][j] += 1
            if j < nY - 1:
                print(counters[i][j])
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondV[i][j].buckets[1])
                counters[i][j] += 1
            if i > 0:
                print(counters[i][j])
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondH[i - 1][j].buckets[0])
                counters[i][j] += 1
            if i < nX - 1:
                print(counters[i][j])
                Link(lattice[i][j].buckets[counters[i][j]],
                     bondH[i][j].buckets[1])
                counters[i][j] += 1

    # Add to Network
    for x in lattice:
        for y in x:
            network.addNode(y)
    for x in onSite:
        for y in x:
            network.addNode(y)
    for x in bondV:
        for y in x:
            network.addNode(y)
    for x in bondH:
        for y in x:
            network.addNode(y)

    return network


def exactIsing2D(J):
    if abs(J) > 1e-4:
        k = 1 / np.sinh(2 * J)**2

        def f(x):
            return np.log(np.cosh(2 * J)**2 + (1 / k) *
                          np.sqrt(1 + k**2 - 2 * k * np.cos(2 * x)))
    else:
        k = np.sinh(2 * J)**2
        def f(x):
            return np.log(np.cosh(2 * J)**2 + 1 - k * np.cos(2 * x))


    inte = quad(f, 0, np.pi)[0]

    return np.log(2) / 2 + (1 / (2 * np.pi)) * inte


def IsingModel3Dopen(nX, nY, nZ, h, J, accuracy):
    network = Network()

    # Place to store the tensors
    lattice = [[[] for j in range(nY)] for i in range(nX)]
    onSite = [[[] for j in range(nY)] for i in range(nX)]
    bondV = [[[] for j in range(nY)] for i in range(nX - 1)]
    bondH = [[[] for j in range(nY - 1)] for i in range(nX)]
    bondZ = [[[] for j in range(nY)] for i in range(nX)]

    # Each lattice site has seven indices of width five, and returns zero if
    # they are unequal and one otherwise.
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                counter = 0
                if i == 0 or i == nX - 1:
                    counter += 1
                if j == 0 or j == nY - 1:
                    counter += 1
                if k == 0 or k == nZ - 1:
                    counter += 1
                lattice[i][j].append(
                    Node(
                        IdentityTensor(
                            2,
                            7 - counter,
                            accuracy=accuracy)))

    # Each on-site term has one index of width two, and returns exp(-h) or
    # exp(h) for 0 or 1 respectively.
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                arr = np.zeros((2))
                arr[0] = np.exp(-h)
                arr[1] = np.exp(h)
                onSite[i][j].append(Node(ArrayTensor(arr)))

    # Each bond term has two indices of width two and returns
    # exp(-J*(1+delta(index0,index1))/2).
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                arr = np.zeros((2, 2))
                arr[0][0] = np.exp(-J)
                arr[1][1] = np.exp(-J)
                arr[0][1] = np.exp(J)
                arr[1][0] = np.exp(J)
                if i < nX - 1:
                    bondV[i][j].append(Node(ArrayTensor(arr)))
                if j < nY - 1:
                    bondH[i][j].append(Node(ArrayTensor(arr)))
                if k < nZ - 1:
                    bondZ[i][j].append(Node(ArrayTensor(arr)))

    # Attach links
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                counter = 0

                Link(
                    lattice[i][j][k].buckets[counter],
                    onSite[i][j][k].buckets[0])
                counter += 1

                if i > 0:
                    Link(lattice[i][j][k].buckets[counter],
                         bondV[i - 1][j][k].buckets[0])
                    counter += 1

                if i < nX - 1:
                    Link(
                        lattice[i][j][k].buckets[counter],
                        bondV[i][j][k].buckets[1])
                    counter += 1

                if j > 0:
                    Link(lattice[i][j][k].buckets[counter],
                         bondH[i][j - 1][k].buckets[0])
                    counter += 1

                if j < nY - 1:
                    Link(
                        lattice[i][j][k].buckets[counter],
                        bondH[i][j][k].buckets[1])
                    counter += 1

                if k > 0:
                    Link(lattice[i][j][k].buckets[counter],
                         bondZ[i][j][k - 1].buckets[0])
                    counter += 1

                if k < nZ - 1:
                    Link(
                        lattice[i][j][k].buckets[counter],
                        bondZ[i][j][k].buckets[1])
                    counter += 1

    # Add to Network
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                network.addNode(lattice[i][j][k])
                network.addNode(onSite[i][j][k])
                if i < nX - 1:
                    network.addNode(bondV[i][j][k])
                if j < nY - 1:
                    network.addNode(bondH[i][j][k])
                if k < nZ - 1:
                    network.addNode(bondZ[i][j][k])

    return network


def IsingSpinGlass(n, J, k, accuracy):
    network = Network()

    nBonds = int(n * k)

    # Place to store the tensors
    lattice = []
    bond = []

    # Determine bond locations
    options = [(i, j) for i in range(n) for j in range(i)]
    choices = np.random.choice(
        list(
            range(
                len(options))),
        size=nBonds,
        replace=False)
    choices = [options[c] for c in choices]

    ranks = [sum([1 for i, j in options if i == q or j == q])
             for q in range(n)]

    for i in range(n):
        lattice.append(Node(IdentityTensor(2, ranks[i], accuracy=accuracy)))

    counters = [0 for _ in range(n)]

    # Attach bonds
    for i, j in options:
        arr = np.zeros((2, 2))
        arr[0][0] = np.exp(-J)
        arr[1][1] = np.exp(-J)
        arr[0][1] = np.exp(J)
        arr[1][0] = np.exp(J)
        bond.append(Node(ArrayTensor(arr)))
        Link(lattice[i].buckets[counters[i]], bond[-1].buckets[0])
        Link(lattice[j].buckets[counters[j]], bond[-1].buckets[1])
        counters[i] += 1
        counters[j] += 1

    # Add to Network
    for n in lattice:
        network.addNode(n)
    for n in bond:
        network.addNode(n)

    return network
