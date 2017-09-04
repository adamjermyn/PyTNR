import numpy as np

from TNRG.Models.isingModel import IsingModel1D, exactIsing1DJ
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import entropyHeuristic

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def ising1DFreeEnergy(nX, h, J, accuracy):
	n = IsingModel1D(nX, h, J, accuracy)
	n = mergeContractor(n, accuracy, entropyHeuristic, optimize=False, merge=False, plot=False, mergeCut=15)
	return np.log(n.array[0])/nX

fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

for J in [-2,-1,-0.5,0,0.5,1,2]:
	size = list(range(2,25))

	h = 0
	accuracy = 1e-3

	res = []

	for s in size:
		f = ising1DFreeEnergy(s, h, J, accuracy)
		res.append((s, f, f - exactIsing1DJ(s, J)))

	res = np.array(res)

	ax1.plot(res[:,0],res[:,1],label='J='+str(J))
	ax1.set_ylabel('Free energy per site')
	ax1.set_xlabel('Number of sites')

	ax2.plot(res[:,0],res[:,2])
	ax2.set_yscale('symlog', linthreshy=1e-16)
	ax2.set_ylabel('Residual')
	ax2.set_xlabel('Number of sites')

ax1.legend()
plt.tight_layout()
plt.savefig('./ising1DJ.pdf')

