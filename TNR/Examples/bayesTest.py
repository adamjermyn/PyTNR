import numpy as np
import time

from TNR.Models.bayes import BayesTest2
from TNR.Contractors.contractor import contractor
from TNR.Contractors.heuristics import utilHeuristic as heuristic

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])
import logging

accuracy = 1e-3

observations = np.random.randint(0, high=20, size=(20, 2))
observations[:, 1] = observations[:, 0] + observations[:, 1]

res = 12

discreteG = np.linspace(0, 1, num=res, endpoint=True)
discreteQ = np.linspace(0, 1, num=res, endpoint=True)
discreteW = np.linspace(0, 1, num=res, endpoint=True)
discreteH = np.linspace(0, 1, num=res, endpoint=True)

n = BayesTest2(
    observations,
    discreteG,
    discreteQ,
    discreteW,
    discreteH,
    accuracy)

c = contractor(n)
done = False
while not done:
    node, done = c.take_step(heuristic)
    eliminateLoops(node.tensor, False)
n = c.network

print(n.nodes.pop().tensor.array)
