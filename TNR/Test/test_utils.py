import numpy as np

import TNR.Utilities.arrays as arrays

epsilon = 1e-10

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
