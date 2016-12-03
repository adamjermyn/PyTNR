import sys 
sys.path.append('../Models/') 
sys.path.append('../Contractors/') 
import numpy as np
from compress import compressLink
from isingModel import IsingModel2D
from mergeContractor import mergeContractor

nX = 10
nY = 10
accuracy = 1e-4
h = 0.1
J = 0.45

n = IsingModel2D(nX, nY, h, J)

n = mergeContractor(n, accuracy, optimize=True, merge=True, verbose=1)



