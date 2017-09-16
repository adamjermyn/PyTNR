import numpy as np

from TNR.Models.isingModel import IsingModel1D, exactIsing1Dh, exactIsing1DJ, IsingModel2D, exactIsing2D
from TNR.Contractors.mergeContractor import mergeContractor

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
            optimize=False,
            merge=False,
            verbose=0)
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
            optimize=False,
            merge=False,
            verbose=0)
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
        n = mergeContractor(n, accuracy, optimize=True, merge=False, verbose=0)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, optimize=True, merge=False, verbose=0)
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
        n = mergeContractor(n, accuracy, optimize=False, merge=True, verbose=0)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, optimize=False, merge=True, verbose=0)
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
        n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=0)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1Dh(h)) < 2 * nX * epsilon

    h = 0.0

    for i in range(5):
        nX = np.random.randint(2, high=10)
        J = np.random.randn(1)
        n = IsingModel1D(nX, h, J, accuracy)
        n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=0)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / nX -
                   exactIsing1DJ(nX, J)) < 2 * nX * epsilon


def test_Ising2D():
    nX = 4
    nY = 4
    accuracy = 1e-5
    h = 0.0

    for i in range(5):
        J = np.random.randn(1)
        n = IsingModel2D(nX, nY, h, J, accuracy)
        n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=0)
        assert len(n.nodes) == 1
        nn = n.nodes.pop()
        assert abs(np.log(nn.tensor.array) / (nX * nY) - exactIsing2D(J)
                   ) < 2 * nX * nY * epsilon + abs(exactIsing2D(J)) / max(nX, nY)
