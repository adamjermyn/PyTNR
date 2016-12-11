import numpy as np

from TNRG.Utilities import utils

def test_ndArrayMatrix():
	x = np.random.randn(2,2,3,3,4,4)

	y = utils.ndArrayToMatrix(x, 2, front=True)

	assert y.shape == (3,2*2*3*4*4)

	z = utils.matrixToNDArray(y, (2,2,3,3,4,4), 2, front=True)

	assert np.sum((x - z)**2) == 0