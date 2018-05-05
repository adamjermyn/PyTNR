import numpy as np

import TNR.Utilities.arrays as arrays

epsilon = 1e-10


def test_insertIndex():
    x = np.random.randn(2, 3, 4, 5)
    y = arrays.insertIndex(x, 1, 2)

    assert y.shape == (2, 4, 3, 5)
    assert np.sum((y - np.swapaxes(x, 1, 2))**2) < epsilon

    z = arrays.insertIndex(y, 1, 2)

    assert z.shape == (2, 3, 4, 5)
    assert np.sum((z - x)**2) < epsilon

    y = arrays.insertIndex(x, 0, 2)
    assert y.shape == (3, 4, 2, 5)

    z = arrays.insertIndex(y, 2, 0)
    assert z.shape == (2, 3, 4, 5)
    assert np.sum((z - x)**2) < epsilon

    y = arrays.insertIndex(x, 0, 3)
    assert y.shape == (3, 4, 5, 2)


def test_ndArrayMatrix():
    x = np.random.randn(2, 2, 3, 3, 4, 4)

    y = arrays.ndArrayToMatrix(x, 2, front=True)

    assert y.shape == (3, 2 * 2 * 3 * 4 * 4)

    z = arrays.matrixToNDArray(y, (2, 2, 3, 3, 4, 4), 2, front=True)

    assert z.shape == x.shape

    assert np.sum((x - z)**2) == 0

    y = arrays.ndArrayToMatrix(x, 2, front=False)

    assert y.shape == (2 * 2 * 3 * 4 * 4, 3)

    z = arrays.matrixToNDArray(y, (2, 2, 3, 3, 4, 4), 2, front=False)

    assert np.sum((x - z)**2) == 0
