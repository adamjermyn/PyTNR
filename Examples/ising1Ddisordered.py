import numpy as np

from TNRG.Models.isingModel import IsingModel1Ddisordered, exactIsing1DJ
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import entropyHeuristic

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def ising1DFreeEnergy(nX, h, J, accuracy):
	n = IsingModel1Ddisordered(nX, h, J, accuracy)
	n = mergeContractor(n, accuracy, entropyHeuristic, optimize=False, merge=False, plot=False, mergeCut=15)
	return np.log(n.array[0])/nX

fig = plt.figure(figsize=(7,7))
ax = plt.subplot(111)

h = 1
J = 1

size = [2,3,4,5,10,15,20,25,30,40,50,60,70,80,90,100,150,200,250,300]
size = 3*size
accuracy = 1e-3
res = []

for s in size:
	print(s)
	f = ising1DFreeEnergy(s, h, J, accuracy)
	res.append((s, f, f - exactIsing1DJ(s, J)))

res = np.array(res)
ax.scatter(res[:,0],res[:,1])
ax.set_ylabel('Free energy per site')
ax.set_xlabel('Number of sites')

plt.tight_layout()
plt.savefig('./ising1Ddisordered.pdf')

