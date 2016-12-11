import numpy as np
from scipy.linalg import expm
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

def test_flatten():
	for i in range(5):
		x = np.random.randn(3,3,5)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))
		assert np.sum((xt.flatten([0,1]).array - np.reshape(x, (-1,5)).T)**2) < epsilon

	for i in range(5):
		x = np.random.randn(2,2,2,2,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))
		assert np.sum((xt.flatten([1,2]).array - np.transpose(np.reshape(x, (2,4,2,2,2)),axes=[0,2,3,4,1]))**2) < epsilon

def test_IndexFactor():
	for i in range(5):
		x = np.random.randn(2,2,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))

		y = np.random.randn(2,2,2,2)
		yt = TreeTensor(accuracy = epsilon)
		yt.addTensor(ArrayTensor(y))

		# Compute inner product
		zt = xt.contract([0],yt,[0])

		# Generate a random unitary matrix
		r = np.random.randn(2,2)
		r += r.T
		u = expm(1j*r)

		# Apply to factors on both x and y
		us = np.sqrt(u)
		factX, indX = xt.getIndexFactor(0)
		factY, indY = yt.getIndexFactor(0)
		factX = np.tensordot(factX, us, axes=([indX],[0]))
		factY = np.tensordot(factY, us, axes=([indY],[1]))
		xt.setIndexFactor(0, factX)
		yt.setIndexFactor(0, factY)
		zt2 = xt.contract([0],yt,[0])

		assert np.sum((zt.array - zt2.array)**2) < epsilon

def test_optimize():
	for i in range(5):
		x = np.random.randn(2,2,2,2)
		xt = TreeTensor(accuracy = epsilon)
		xt.addTensor(ArrayTensor(x))

		y = np.random.randn(2,2,2,2)
		yt = TreeTensor(accuracy = epsilon)
		yt.addTensor(ArrayTensor(y))

		zt = xt.contract([0],yt,[0])

		arr1 = zt.array

		zt.optimize()

		arr2 = zt.array

		assert np.sum((arr1 - arr2)**2) < epsilon


