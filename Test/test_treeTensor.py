import numpy as np
from TNRG.TreeTensor.treeTensor import TreeTensor
from TNRG.Tensor.arrayTensor import ArrayTensor

epsilon = 1e-10

def test_init():
	for i in range(5):
		x = np.random.randn(3,3,4)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))

		assert xt.shape == (3,3,4)
		assert xt.rank == 3
		assert xt.size == 3*3*4

		assert np.sum((xt.array - x)**2) < epsilon

	for i in range(5):
		x = np.random.randn(2,2,2,2,2,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))

		assert xt.shape == (2,2,2,2,2,2,2)
		assert xt.rank == 7
		assert xt.size == 2**7

		assert np.sum((xt.array - x)**2) < epsilon

def test_contract():
	for i in range(5):
		x = np.random.randn(2,2,2,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))

		y = np.random.randn(2,2,2,2,2)
		yt = TreeTensor(accuracy = epsilon)
		yt.addTensor(ArrayTensor(y))

		zt = xt.contract([0,1,4],yt,[2,3,4])
		assert np.sum((zt.array - np.einsum('ijklm,qwijm->klqw',x,y))**2) < epsilon

		zt = yt.contract([0,1,4],xt,[2,3,4])
		assert np.sum((zt.array - np.einsum('ijklm,qwijm->klqw',y,x))**2) < epsilon

	for i in range(5):
		x = np.random.randn(2,2,3,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))

		y = np.random.randn(3,2,2,2,2)
		yt = TreeTensor(accuracy = epsilon)
		yt.addTensor(ArrayTensor(y))

		zt = xt.contract([0,1,4],yt,[2,3,4])
		assert np.sum((zt.array - np.einsum('ijklm,qwijm->klqw',x,y))**2) < epsilon

def test_trace():
	for i in range(5):
		x = np.random.randn(2,2,2,2,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))
		assert np.sum((xt.trace([0],[1]).array - np.einsum('iijklm->jklm',x))**2) < epsilon
		assert np.sum((xt.trace([0,2],[5,4]).array - np.einsum('ijklki->jl',x))**2) < epsilon

'''
def test_flatten():
	for i in range(5):
		x = np.random.randn(3,3,5)
		xt = ArrayTensor(x)
		assert np.sum((xt.flatten([0,1]).array - np.reshape(x, (-1,5)).T)**2) < epsilon

def test_getIndexFactor():
	for i in range(5):
		x = np.random.randn(3,3,3)
		xT = ArrayTensor(x)
		a, j = xT.getIndexFactor(0)
		assert np.sum((x/np.max(np.abs(x)) - a)**2) < epsilon
		assert j == 0

def test_setIndexFactor():
	for i in range(5):
		x = np.random.randn(3,3,3)
		xT = ArrayTensor(x)
		y = np.random.randn(3,3,3)
		assert np.sum((xT.setIndexFactor(0, y).array - y*np.exp(xT.logScalar))**2) < epsilon
'''