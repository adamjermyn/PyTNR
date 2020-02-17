import networkx
from copy import deepcopy

def eliminateLoops(network, return_copy):
    if return_copy:
        network = deepcopy(network)

    tm = traceMin(network)

    while len(networkx.cycles.cycle_basis(network.toGraph())) > 0:

        logger.debug('Cycle utility is ' +
                     str(tm.util) +
                     ' and there are ' +
                     str(len(networkx.cycles.cycle_basis(network.toGraph()))) +
                     ' cycles remaining.')

        merged = tm.mergeSmall()

        if not merged:
            best = tm.bestSwap()
            tm.swap(best[1], best[2], best[3])

        logger.debug(str(network))

    counter0 += 1
    assert len(networkx.cycles.cycle_basis(network.toGraph())) == 0

    return network