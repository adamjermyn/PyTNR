import numpy as np
import time

from TNRG.Models.isingModel import IsingModel2D, exactIsing2D
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import loopHeuristic as heuristic

from TNRG.Utilities.logger import makeLogger
from TNRG import config
logger = makeLogger(__name__, config.levels['generic'])
import logging


def ising2DFreeEnergy(nX, nY, h, J, accuracy):
	n = IsingModel2D(nX, nY, h, J, accuracy)
	n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=False, plot=False)
	return n.array[1]/(nX*nY)

size = (7,7)
jran = np.linspace(-3,3,num=35,endpoint=True)

res = []
for J in jran:
	h = 0
	accuracy = 1e-3

	logger.info('Examining system of J = ' + str(J) + '.')
	start = time.clock()
	f = ising2DFreeEnergy(size[0], size[1], h, J, accuracy)
	end = time.clock()
	res.append((J, f, f - exactIsing2D(J), end - start))

res = np.array(res)

print('Result:')
print(res)

np.savetxt('ising2DJ_sweep.dat', res)
