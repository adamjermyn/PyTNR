import numpy as np

from TNR.Models.peps import *
from TNR.Contractors.mergeAllContractor import mergeContractor
import sys

nn = int(sys.argv[1])
nX = nn
nY = nn

accuracy = 1e-3

if sys.argv[2] == 'akltd':
    A, B = aklt2d()
elif sys.argv[2] == 'bosonic_insulator':
    A, B = featureless_bosonic_insulator()
elif sys.argv[2] == 'hardcore_bosonic':
    A, B = featureless_bosonic_insulator_hardcore()
elif sys.argv[2] == 'featureless_su2':
    A, B = featureless_su2()
else:
    exit()

n = peps2Dhoneycomb(nX, nY, A, B, accuracy)
n = mergeContractor(n, accuracy, optimize=False)


for nn in n.nodes:
    print('F/N: ', nn.tensor.logNorm / (nX * nY))
