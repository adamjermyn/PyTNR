import networkx
from copy import deepcopy
from TNR.Network.traceMin import traceMin

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['traceMin'])

def swap_elim(tensor, return_copy):
    if return_copy:
        tensor = deepcopy(tensor)

    network = tensor.network
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

    assert len(networkx.cycles.cycle_basis(network.toGraph())) == 0

    return tensor

def swap_elim_node(node, return_copy):
    if return_copy:
        node = deepcopy(node)

    node.tensor = swap_elim(node.tensor, False)

    return node

def swap_elim_network(network, node, return_copy):
    if return_copy:
        network = deepcopy(network)
        node = list(n for n in network.nodes if n.id == node.id)[0]

    return network, swap_elim_node(node, False)