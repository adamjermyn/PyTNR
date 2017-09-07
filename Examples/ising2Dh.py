import numpy as np
import time

from TNRG.Models.isingModel import IsingModel2D, exactIsing1Dh
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import entropyHeuristic

def ising2DFreeEnergy(nX, nY, h, J, accuracy):
	n = IsingModel2D(nX, nY, h, J, accuracy)
	n = mergeContractor(n, accuracy, entropyHeuristic, optimize=True, merge=False, plot=False)
	arr = n.array
	return np.log(n.array[0])/(nX*nY)

for h in [-2,-1,-0.5,0,0.5,1,2]:
	size = [(2,2),(2,3),(2,4),(3,3),(2,5),(3,4),(4,4),(3,6),(4,5),(3,7),(3,8),(5,5),(3,9),(4,7),(5,6),(4,8),(5,7),(6,6)]

	J = 0
	accuracy = 1e-5

	res = []

	for s in size:
		start = time.clock()
		f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
		end = time.clock()
		res.append((s[0]*s[1], f, f - exactIsing1Dh(h), end - start))

	res = np.array(res)

	np.savetxt('ising1Dh_h='+str(h)+'.dat', res)
