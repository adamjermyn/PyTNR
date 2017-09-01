import numpy as np
from TNRG.Models.isingModel import IsingModel2Dopen
from TNRG.Contractors.mergeContractor import mergeContractor

fi = open('out.txt', 'w+')

s = 14

nX = s
nY = s

accuracy = 1e-3

h = 0.1
J = 0.5

n = IsingModel2Dopen(nX, nY, h, J, accuracy)
n = mergeContractor(n, accuracy, optimize=False, merge=False, mergeCut=10, verbose=2)

assert len(n.nodes) == 1

fi.write(str(s) + ',' + str(np.log(next(iter(n.nodes)).tensor.array)/s**2)+'\n')
fi.flush()