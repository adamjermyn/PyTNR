import numpy as np

from TNRG.Models.isingModel import IsingModel2D, exactIsing1Dh
from TNRG.Contractors.mergeContractor import mergeContractor, entropyHeuristic

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def ising2DFreeEnergy(nX, nY, h, J, accuracy):
	n = IsingModel2D(nX, nY, h, J, accuracy)
	n = mergeContractor(n, accuracy, entropyHeuristic, optimize=True, merge=False, plot=False, mergeCut=15, verbose=0)
	arr = n.array
	return np.log(n.array[0])/(nX*nY)


fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

for h in [-2,-1,-0.5,0,0.5,1,2]:
	size = [(2,2),(2,3),(2,4),(3,3),(2,5),(3,4),(4,4),(3,6),(4,5),(3,7),(3,8),(5,5),(3,9),(4,7),(5,6),(4,8),(5,7),(6,6)]

	J = 0
	accuracy = 1e-5

	res = []

	for s in size:
		f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
		res.append((s[0]*s[1], f, f - exactIsing1Dh(h)))

	res = np.array(res)

	ax1.plot(res[:,0],res[:,1],label='h='+str(h))
	ax1.set_ylabel('Free energy per site')
	ax1.set_xlabel('Number of sites')

	ax2.plot(res[:,0],res[:,2])
	ax2.set_yscale('symlog', linthreshy=1e-16)
	ax2.set_ylabel('Residual')
	ax2.set_xlabel('Number of sites')

ax1.legend()
plt.tight_layout()
plt.savefig('./ising2Dh.pdf')

