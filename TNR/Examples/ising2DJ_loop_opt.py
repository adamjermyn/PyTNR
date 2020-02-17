import numpy as np
import time
import networkx

from TNR.Models.isingModel import IsingModel2Ddisordered
from TNR.Contractors.contractor import replicaContractor
from TNR.Contractors.heuristics import loopHeuristic as heuristic

from TNR.Actions.loop_svd_elim import loop_svd_single_elim_network as eliminateLoops
from TNR.Actions.optimize_loop import loop_svd_optimize_network as optimize
from TNR.Actions.basic_actions import merge_all_nodes

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['generic'])


def ising2DFreeEnergy(nX, nY, h, J, accuracy):
    n = IsingModel2Ddisordered(nX, nY, h, J, accuracy)

    # Merge all
    n = merge_all_nodes(n, False)

    # Eliminate loops
    c = replicaContractor(n, 5, 1e6)
    ind = 0
    nodes = list(n.nodes)
    for node in nodes:
        done = False
        while len(networkx.cycles.cycle_basis(node.tensor.network.toGraph())) > 0:
            ind = c.index_of_least_cost()
            next_info, replaced = c.perform_action(ind, eliminateLoops, node, False)

    n = c.replicas[ind].network

    arr, log_arr, bdict = n.array
    return (np.log(np.abs(arr)) + log_arr) / (nX * nY)


h = 1
J = 1
accuracy = 1e-6
size = [(2, 2), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4), (4, 4), (3, 6), (4, 5), (3, 7), (3, 8), (5, 5), (3, 9),
        (4, 7), (5, 6), (4, 8), (5, 7), (6, 6), (6, 7), (7, 7), (7, 8), (8, 8), (8, 9)]
res = []

for s in size:
    for _ in range(3):
        logger.info(
            'Examining system of size ' +
            str(s) +
            ' and J = ' +
            str(J) +
            '.')
        start = time.clock()
        f = ising2DFreeEnergy(s[0], s[1], h, J, accuracy)
        end = time.clock()
        res.append((s[0] * s[1], f, end - start))

res = np.array(res)

print(res)

np.savetxt('ising2D_disordered.dat', res)
