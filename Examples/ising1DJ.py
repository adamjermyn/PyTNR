import numpy as np
import time

from TNR.Models.isingModel import IsingModel1D, exactIsing1DJ
from TNR.Contractors.mergeContractor import mergeContractor
from TNR.Contractors.heuristics import entropyHeuristic

def ising1DFreeEnergy(nX, h, J, accuracy):
	n = IsingModel1D(nX, h, J, accuracy)
	n = mergeContractor(n, accuracy, entropyHeuristic, optimize=False, merge=False, plot=False)
	return n.array[1]/nX

for J in [-2,-1,-0.5,0,0.5,1,2]:
	size = list(range(2,25))

	h = 0
	accuracy = 1e-3

	res = []

	for s in size:
		start = time.clock()
		f = ising1DFreeEnergy(s, h, J, accuracy)
		end = time.clock()
		res.append((s, f, f - exactIsing1DJ(s, J), end - start))

	res = np.array(res)

	np.savetxt('ising1DJ_J='+str(J)+'.dat', res)

