import sys
sys.path.append('../TensorNetwork/')
from network import Network
import numpy as np
import cProfile

network = Network()

n = 100
nB = 5

links = []

# Temporary counters
counters = [0 for i in range(n)]

netBonds = 0

# Figure out bonds
for i in range(20 * n * nB):
    i, j = np.random.randint(0, n, size=2)
    if i != j and counters[i] < nB and counters[j] < nB:
        counters[i] += 1
        counters[j] += 1
        netBonds += 1
        links.append((i, j, np.random.randn()))


def glass(temp):
    # Place to store the tensors
    lattice = []
    bonds = []

    # Lattice tensors
    for i in range(n):
        arr = np.zeros(tuple(2 for j in range(counters[i])))
        np.fill_diagonal(arr, 1.0)
        lattice.append(network.addNodeFromArray(arr))

    # Temporary counters
    counters2 = [0 for i in range(n)]

    for i in range(len(links)):
        arr = np.zeros((2, 2))
        arr[0][0] = np.exp(-links[i][2] / temp)
        arr[1][1] = np.exp(-links[i][2] / temp)
        arr[0][1] = np.exp(links[i][2] / temp)
        arr[1][0] = np.exp(links[i][2] / temp)
        bonds.append(network.addNodeFromArray(np.copy(arr)))
        lattice[links[i][0]].addLink(bonds[-1], counters2[links[i][0]], 0)
        lattice[links[i][1]].addLink(bonds[-1], counters2[links[i][1]], 1)
        counters2[links[i][0]] += 1
        counters2[links[i][1]] += 1

    network.trace()

    while len(network.topLevelLinks()) > 0:
        network.merge(mergeL=True, compress=True)

        print len(network.topLevelNodes()), network.topLevelSize(), np.product(network.largestTopLevelTensor()), network.largestTopLevelTensor()
        sizes = [nn.tensor().array().size for nn in network.topLevelNodes()]
        print sizes, np.prod(np.array(sizes, dtype=float))

    return np.log(list(network.topLevelNodes())[0].tensor().array())


cProfile.run('glass(1.0)')
print netBonds
