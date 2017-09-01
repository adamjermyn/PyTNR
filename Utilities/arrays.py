import numpy as np

def insertIndex(arr, ind, newInd):
	'''
	This method removes the specified index (ind) and inserts
	it in the new location (newInd).
	'''
	perm = list(range(len(arr.shape)))
	perm.remove(ind)
	perm.insert(newInd, ind)
	arr = np.transpose(arr, axes=perm)
	return arr

def permuteIndices(arr, indices, front=True):
	'''
	This method moves the indices specified in indices
	to be the first ones in the array in the order in which they appear in indices.
	If front is False it instead moves them to be the last ones.
	'''
	shape = arr.shape
	perm = list(range(len(shape)))

	for i in indices:
		perm.remove(i)
	for j,i in enumerate(indices):
		if front:
			perm.insert(j, i)
		else:
			perm.insert(len(shape)-len(indices) + j, i)
	return np.transpose(arr, axes=perm)

def ndArrayToMatrix(arr, index, front=True):
		'''
		This method flattens the array along all indices other than
		index and does so in a way which preserves the ordering of the other
		axes when unflattened.

		This method also takes as input a boolean variable front. If front is True
		then the special index is pushed to the beginning. If front is False then the
		special index is pushed to the back.
		'''
		arr = insertIndex(arr, index, 0)
		arr = np.reshape(arr, (arr.shape[0],-1))

		if not front:
			arr = np.transpose(arr)

		return arr

def matrixToNDArray(matrix, shape, index, front=True):
		'''
		This method takes a 2D array and reshapes it to the given shape.
		The reshape operation only modifies one of the axes of the matrix.
		This is either the first (front) or last (not front) depending on the
		boolean variable front. Whichever index is not reshaped is then
		put in the position specified by index.

		This method is meant to be the inverse of ndArrayToMatrix.
		'''
		if not front:
			matrix = np.transpose(matrix)

		shm = shape[:index] + shape[index+1:]

		matrix = np.reshape(matrix, [shape[index]] + list(shm))

		perm = list(range(len(shape)))
		perm = perm[1:]
		perm.insert(index, 0)

		matrix = np.transpose(matrix, axes=perm)

		return matrix