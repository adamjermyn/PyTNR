import numpy as np
import gc

from TNRG.Models.isingModel import IsingModel2D, exactIsing2D
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.mergeContractor import loopHeuristic as heuristic

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def ising2DFreeEnergy(nX, nY, h, J, accuracy):
	n = IsingModel2D(nX, nY, h, J, accuracy)
	n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=False, plot=True, mergeCut=15, verbose=1)
	return np.log(n.array[0])/(nX*nY), n.array

size = [(2,2),(2,3),(2,4),(3,3),(2,5),(3,4),(4,4),(3,6),(4,5),(3,7),(3,8),(5,5),(3,9),(4,7),(5,6),(4,8),(5,7),(6,6),(6,7),(7,7)]#]#,(7,8),(8,8),(8,9)]#,(9,9),(9,10),(10,10)]

fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)


for J in [-2,-1,0,1,2]:

	print('JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ',J)

	h = 0
	accuracy = 1e-3

	res = []

	for s in size:
		print('JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ',J)
		print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS',s)
		f, arr = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
		res.append((s[0]*s[1], f, f - exactIsing2D(J)))

		gc.collect()
		print(gc.garbage)

	res = np.array(res)

	print(res)

	ax1.plot(res[:,0],res[:,1],label='J='+str(J))
	ax2.plot(res[:,0],res[:,2])


ax1.set_ylabel('Free energy per site')
ax1.set_xlabel('Number of sites')
ax2.set_ylabel('Residual')
ax2.set_xlabel('Number of sites')

ax1.legend()
plt.tight_layout()
plt.savefig('./ising2DJ.pdf')



