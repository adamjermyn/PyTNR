from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Network.traceMin import traceMin
import numpy as np
import networkx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
import sys

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['mergeContractor'])



def managedContractor(n, num_copies, accuracy, heuristic, optimize=True, cost_cap=None):

    networks = list(deepcopy(n) for _ in range(num_copies))

    costs = list(net.compressedSize for net in networks)

    done = False
    while not done:

        # Work with the lowest-cost network
        ind = costs.index(min(costs))
        net = networks[ind]

        # Perform one contraction step
        try:
            logger.info('Performing contraction on network ' + str(ind) + ' with cost ' + str(costs[ind]) + '. Costs: ' + str(costs) + '.')
            logger.info('Network has ' + str(len(net.nodes)) + ' nodes.')
            q, n1, n2 = heuristic(net)
            n3 = net.mergeNodes(n1, n2)

            n3.eliminateLoops()

            if optimize:
                n3.tensor.optimize()

            logger.info('Merging nodes...')
            nn = n3
            if hasattr(
                    nn.tensor, 'compressedSize'):
                done2 = False
                while not done2:
                    merged = net.mergeClosestLinks(
                        n3, compress=True, accuracy=accuracy)
                    if merged is not None:
                        nn.eliminateLoops()
                        merged.eliminateLoops()
                        if optimize:
                            nn.tensor.optimize()
                            merged.tensor.optimize()
                    else:
                        done2 = True

            logger.info('Merging complete.')

            costs[ind] = net.compressedSize

            if len(net.internalBuckets) == 0:
                done = True



        except KeyboardInterrupt:
            exit()

        if cost_cap is not None:
            for i in range(num_copies):
                if costs[i] > cost_cap:
                    logger.info('Network ' + str(i) + ' has exceeded the cost cap. Replacing it with a clone of the best network.')
                    costs[i] = min(costs)
                    networks[i] = deepcopy(networks[costs.index(min(costs))])

    return networks[ind]