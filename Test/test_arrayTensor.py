import numpy as np
from scipy.linalg import expm
from TNRG.Tensor.arrayTensor import ArrayTensor

epsilon = 1e-10

def test_init():
	for i in range(5):
		x = np.random.randn(3,3,4)
		xt = ArrayTensor(x, logScalar=0)

		assert xt.shape == (3,3,4)
		assert xt.rank == 3
		assert xt.size == 3*3*4

		assert np.sum((xt.array - x)**2) < epsilon
		assert xt.logScalar == np.log(np.max(np.abs(x)))
		assert np.sum((xt.scaledArray - x/np.exp(xt.logScalar))**2) < epsilon

		xt2 = ArrayTensor(x, logScalar=1.44)
		assert xt2.logScalar == xt.logScalar + 1.44

def test_contract():
	for i in range(5):
		x = np.random.randn(3,4,3)
		y = np.random.randn(3,4,3)

		xt = ArrayTensor(x)
		yt = ArrayTensor(y)

		zt = xt.contract(0,yt,0)
		assert np.sum((zt.array - np.einsum('ijk,ilm->jklm',x,y))**2) < epsilon

		zt = xt.contract(1,yt,1)
		assert np.sum((zt.array - np.einsum('jik,lim->jklm',x,y))**2) < epsilon

		zt = xt.contract(0,yt,2)
		assert np.sum((zt.array - np.einsum('ijk,lmi->jklm',x,y))**2) < epsilon

def test_trace():
	for i in range(5):
		x = np.random.randn(3,3,5)
		xt = ArrayTensor(x)
		assert np.sum((xt.trace([0],[1]).array - np.einsum('iij->j',x))**2) < epsilon

	for i in range(5):
		x = np.random.randn(3,3,5,4,4)
		xt = ArrayTensor(x)
		assert np.sum((xt.trace([0,3],[1,4]).array - np.einsum('iijkk->j',x))**2) < epsilon

def test_flatten():
	for i in range(5):
		x = np.random.randn(3,3,5)
		xt = ArrayTensor(x)
		assert np.sum((xt.flatten([0,1]).array - np.reshape(x, (-1,5)).T)**2) < epsilon

	for i in range(5):
		x = np.random.randn(3,3,5)
		xt = ArrayTensor(x)
		assert np.sum((xt.flatten([1,0]).array - np.reshape(np.swapaxes(x,0,1), (-1,5)).T)**2) < epsilon

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

def test_IndexFactor():
	for i in range(5):
		x = np.random.randn(3,4,5,6)
		xt = ArrayTensor(x)

		y = np.random.randn(3,4,5,6)
		yt = ArrayTensor(y)

		# Compute inner product
		zt = xt.contract([0],yt,[0])

		# Generate a random unitary matrix
		r = np.random.randn(3,3)
		r += r.T
		u = expm(1j*r)

		assert np.sum((np.identity(3) - np.dot(u,np.conjugate(u.T)))**2) < epsilon

		# Apply to factors on both x and y
		factX, indX = xt.getIndexFactor(0)
		factY, indY = yt.getIndexFactor(0)

		factX = np.tensordot(factX, u, axes=([indX],[0]))
		factY = np.tensordot(factY, np.conjugate(u.T), axes=([indY],[0]))

		factX = np.transpose(factX,axes=[3,0,1,2])
		factY = np.transpose(factY,axes=[3,0,1,2])

		xt = xt.setIndexFactor(0, factX)
		yt = yt.setIndexFactor(0, factY)

		assert xt.shape == (3,4,5,6)
		assert yt.shape == (3,4,5,6)

		zt2 = xt.contract([0],yt,[0])

		assert np.sum((zt.array - zt2.array)**2) < epsilon
