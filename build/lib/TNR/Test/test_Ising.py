import numpy as np

from TNR.Models.isingModel import IsingModel1D, exactIsing1Dh, exactIsing1DJ, IsingModel2D, exactIsing2D
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

epsilon = 1e-10


def test_Ising1D():
    accuracy = epsilon

    J = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        h = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(
            n,
            accuracy,
            heuristic,
            optimize=False,
            merge=False)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(
            n,
            accuracy,
            heuristic,
            optimize=False,
            merge=False)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1DJ(nX, J)) < 2 * nX * epsilon


def test_Ising1D_Opt():
    accuracy = epsilon

    J = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        h = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=False)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=False)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1DJ(nX, J)) < 2 * nX * epsilon


def test_Ising1D_Merge():
    accuracy = epsilon

    J = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        h = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=False, merge=True)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=False, merge=True)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1DJ(nX, J)) < 2 * nX * epsilon


def test_Ising1D_Opt_Merge():
    accuracy = epsilon

    J = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        h = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=True)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=True)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1DJ(nX, J)) < 2 * nX * epsilon

def test_Ising2D_No_Opt():
    nX = 3
    nY = 3
    accuracy = 1e-3
    h = 0.0

    for i in range(5):
        J = np.random.randn(1)
        n = IsingModel2D(nX, nY, h, J, accuracy)
        n = mergeContractor(n, accuracy, heuristic, optimize=False, merge=False, plot=False)
        arr, lg = n.array[:2]
        if len(arr.shape) > 0:
            lg = lg + np.log(sum(arr))
        print(lg)
        assert abs(lg / (nX * nY) - exactIsing2D(J)
                   ) < 2 * nX * nY * epsilon + abs(exactIsing2D(J)) / max(nX, nY)


