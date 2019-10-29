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
            q, n1, n2 = heuristic(net)
            n3 = net.mergeNodes(n1, n2)

            n3.eliminateLoops()

            if optimize:
                n3.tensor.optimize()

            costs[ind] = net.compressedSize

            if len(net.internalBuckets) == 0:
                done = True
        except KeyboardInterrupt:
            exit()
        except:
            e = sys.exc_info()[0]
            logger.info(str(e))
            logger.info('Failed to contract network ' + str(ind) + '.')
            logger.info('Replacing that with a clone of the next best network.')
            # Clone the current best network in place of the failed one
            del networks[ind]
            del costs[ind]
            ind = costs.index(min(costs))
            networks.append(deepcopy(networks[ind]))
            costs.append(costs[ind])

        if cost_cap is not None:
            for i in range(num_copies):
                if costs[i] > cost_cap:
                    logger.info('Network ' + str(i) + ' has exceeded the cost cap. Replacing it with a clone of the best network.')
                    costs[i] = min(costs)
                    networks[i] = deepcopy(networks[costs.index(min(costs))])

    return networks[ind]