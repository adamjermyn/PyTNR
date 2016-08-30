import numpy as np

def kroneckerDelta(dim, length):
	if dim == 0:
		return 1
	elif dim == 1:
		return np.ones(length)
	elif dim > 1:
		arr = np.zeros(tuple(length for i in range(dim)))
		np.fill_diagonal(arr,1.0)
		return arr