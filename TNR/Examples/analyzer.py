import numpy as np
import time

from TNR.Models.isingModel import IsingModel2DdisorderedProbs
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])

import pickle
n, buckets = pickle.load(open('2DIsingDisorderedProbs.net', 'rb'))

n = n.nodes.pop()
t = n.tensor

indices = []

for i in range(len(buckets)):
    indices.append([])
    for j in range(len(buckets[i])):
        indices[i].append(n.buckets.index(buckets[i][j]))

def traceOut(tens, inds):
	while tens.rank > len(inds):
		i = np.random.randint(tens.rank)
		if i not in inds:
			tens = tens.traceOut(i)
			for j in range(len(inds)):
				if inds[j] > i:
					inds[j] -= 1
	return tens

def nPoint(arr):
	print(arr)
	acc = np.sum(arr)
	arr /= acc
	for i in range(len(arr.shape)):
		sl = list([slice(0, arr.shape[k]) for k in range(i)]) + [0] + list([slice(0, arr.shape[k]) for k in range(i+1, len(arr.shape))])
		arr[sl] *= 1
		sl = list([slice(0, arr.shape[k]) for k in range(i)]) + [1] + list([slice(0, arr.shape[k]) for k in range(i+1, len(arr.shape))])
		arr[sl] *= -1

	print(arr)

	return np.sum(arr)

twop = []

for i in range(len(buckets)):
    twop.append([])
    for j in range(len(buckets[i])):
        if i == 0 and j == 0:
            twop[i].append(np.identity(2))
        else:
            twop[i].append(traceOut(t, [indices[0][0], indices[i][j]]).array)
        twop[i][j] = nPoint(twop[i][j])
        print('-----')
        print(i,j)
        print(twop[i][-1])

twopoint = np.array(twop)

import matplotlib.pyplot as plt

plt.imshow(twopoint)
plt.colorbar()
plt.show()

x = [0, 1, 4, 2, 1, 3, 2]
y = [0, 1, 1, 2, 3, 2, 1]
inds = list([indices[x[i]][y[i]] for i in range(len(x))])

twop = []

for i in range(len(buckets)):
    twop.append([])
    for j in range(len(buckets[i])):
        if i == 0 and j == 0:
            twop[i].append(np.identity(2))
        else:
            twop[i].append(traceOut(t, inds + [indices[i][j]]).array)
        twop[i][j] = nPoint(twop[i][j])
        print('-----')
        print(i,j)
        print(twop[i][-1])

twopoint = np.array(twop)

import matplotlib.pyplot as plt

plt.imshow(twopoint)
plt.colorbar()
plt.show()