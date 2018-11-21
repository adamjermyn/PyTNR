import numpy as np

from TNR.Utilities.switches import switch_tree, binary_switch
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Utilities.graphPlotter import plot, plotGraph
from TNR.Utilities.linalg import L2error

def test_switches():

	accuracy = 1e-10

	arr1 = np.array([1,2], dtype=np.float64)
	arr2 = np.array([3,4], dtype=np.float64)
	arr1 /= np.sum(arr1)
	arr2 /= np.sum(arr2)

	t1 = ArrayTensor(arr1)
	t2 = ArrayTensor(arr2)

	b = binary_switch(accuracy, 2, 2)

	b = b.contract([2], t1, [0])
	b = b.contract([2], t2, [0])

	arr = b.array
	arr /= np.sum(arr, axis=1)

	assert L2error(arr[0], arr1) < accuracy
	assert L2error(arr[1], arr2) < accuracy


def test_multimarge():
	accuracy = 1e-10

	tensors = list(ArrayTensor(1+np.arange(i)) for i in range(1,8))

	tree, N = switch_tree(accuracy, list(t.shape[0] for t in tensors))

	print(tree.shape)

	for t in tensors:
		tree = tree.contract([N+1], t, [0], elimLoops=False)
		print(tree.shape)


	bits = [1,1,0]

	bits = list(ArrayTensor(np.array([0,1])) if b==1 else ArrayTensor(np.array([1,0])) for b in bits)

	for b in bits:
		tree = tree.contract([0], b, [0], elimLoops=False)

	tree.network.contractRank2()
	plot(tree.network, fname='test.pdf')
	print(tree.shape)
	print(tree.array / tree.array[0])

	assert False
	#print(tree.array)