'''
A simple script for profiling different SVD options on your machine.
'''

import numpy as np
from scipy.linalg.interpolative import svd as svdI
from scipy.sparse.linalg import svds
from time import perf_counter

def time(str):
	start = perf_counter()
	eval(str)
	end = perf_counter()
	return end - start

Ns = [1, 3, 10, 30, 100, 300, 1000, 3000]
fracs = [0.001, 0.01, 0.1, 0.3]

dense = []
sparse = []
interpolative = []

for N in Ns:
	x = np.random.randn(N, N)
	u, s, v = np.linalg.svd(x)

	dense.append(time('np.linalg.svd(x)'))
	print('Dense', N**2, dense[-1])

	sparse.append([])
	interpolative.append([])
	for f in fracs:
		k = int(N * f)
		if k > 0 and k < N - 3:
			ss = np.array(s)
			ss = ss / np.sum(ss**2)**0.5
			precision = ss[k]
			ss[k:] = 0
			xI = np.einsum('ij,j,jk->ik',u,ss,v)
			sparse[-1].append(time('svds(x, k, which=\'LM\')'))
			print('Sparse', N**2, f, sparse[-1][-1])
			interpolative[-1].append(time('svdI(xI, precision)'))
			print('Interpolative', N**2, f, interpolative[-1][-1])
		else:
			sparse[-1].append(0)
			interpolative[-1].append(0)

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.plot(Ns, dense, label='dense', c='k')

#for i,f in enumerate(fracs):
#	plt.plot(Ns, list(sparse[j][i] for j in range(len(sparse))), label='sparse, f=' + str(f), c=colors[i])

for i,f in enumerate(fracs):
	plt.plot(Ns, list(interpolative[j][i] for j in range(len(sparse))), label='interp, f=' + str(f), linestyle='--', c=colors[i])

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()