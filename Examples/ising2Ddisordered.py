import numpy as np

from TNRG.Models.isingModel import IsingModel2Ddisordered, exactIsing2D
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import loopHeuristic as heuristic

from TNRG.Utilities.logger import makeLogger
from TNRG import config
logger = makeLogger(__name__, config.levels['generic'])

def ising2DFreeEnergy(nX, nY, h, J, accuracy):
	n = IsingModel2Ddisordered(nX, nY, h, J, accuracy)
	n = mergeContractor(n, accuracy, heuristic, optimize=True, merge=False, plot=True, mergeCut=15)
	return n.array[1]/(nX*nY)


import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(111)

h = 1
J = 1
accuracy = 1e-3
size = [(2,2),(2,3),(2,4),(3,3),(2,5),(3,4),(4,4),(3,6),(4,5),(3,7),(3,8),(5,5),(3,9),(4,7),(5,6),(4,8),(5,7),(6,6),(6,7),(7,7),(7,8),(8,8)]#]#,(8,9)]#,(9,9),(9,10),(10,10)]

res = []

for s in size:
	for _ in range(3):
		logger.info('Examining system of size ' + str(s) + ' and J = ' + str(J) + '.')
		f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
		res.append((s[0]*s[1], f, f - exactIsing2D(J)))

res = np.array(res)

print(res)

ax1.scatter(res[:,0],res[:,1])

ax1.set_ylabel('Free energy per site')
ax1.set_xlabel('Number of sites')

ax1.legend()
plt.tight_layout()
plt.savefig('./ising2DJdisordered.pdf')



