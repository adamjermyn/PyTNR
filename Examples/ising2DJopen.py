import numpy as np
import time

from TNRG.Models.isingModel import IsingModel2Dopen, exactIsing2D
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import loopHeuristic as heuristic

from TNRG.Utilities.logger import makeLogger
from TNRG import config
logger = makeLogger(__name__, config.levels['generic'])

def ising2DFreeEnergy(nX, nY, h, J, accuracy):
	n = IsingModel2Dopen(nX, nY, h, J, accuracy)
	n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=False, plot=False)
	return n.array[1]/(nX*nY)

for J in [-2,-1,-0.5,0,0.5,1,2]:

	h = 0
	accuracy = 1e-3
	size = [(2,2),(2,3),(2,4),(3,3),(2,5),(3,4),(4,4),(3,6),(4,5),(3,7),(3,8),(5,5),(3,9),(4,7),(5,6),(4,8),(5,7),(6,6),(6,7),(7,7),(7,8),(8,8)]#]#,(8,9)]#,(9,9),(9,10),(10,10)]

	res = []

	for s in size:
		logger.info('Examining system of size ' + str(s) + ' and J = ' + str(J) + '.')
		start = time.clock()
		f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
		end = time.clock()
		res.append((s[0]*s[1], f, f - exactIsing2D(J), end - start))

	res = np.array(res)

	print(res)

	np.savetxt('ising2DJ_open_J='+str(J)+'.dat', res)


