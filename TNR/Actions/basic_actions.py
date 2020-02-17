from copy import deepcopy
from TNR.Contractors.heuristics import simple_heuristic

from TNR.Utilities.logger import makeLogger
import TNR.config as config
logger = makeLogger(__name__, config.levels['basic_actions'])

def merge_nodes(network, heuristic, return_copy):
	'''
	Uses a heuristic to pick the next contraction and performs it.

	Returns the resulting node as well as whether or not the network is fully contracted.
	'''

	if return_copy:
		network = deepcopy(network)

	# Pre-contraction output
	logger.info('Network has ' +
				str(len(network.nodes)) +
				' nodes.')

	# Contraction
	utility, node1, node2 = heuristic(network)
	new_node = network.mergeNodes(node1, node2)

	# Post-contraction output
	logger.info('Contraction had utility ' +
				str(utility) +
				' and resulted in a tensor connected to ' +
				str(len(new_node.connectedNodes)) +
				' nodes.')

	done = False
	if len(network.internalBuckets) == 0:
		done = True

	return network, new_node, done

def merge_all_nodes(n, return_copy):
    '''
    Merges all nodes in an unspecified order.
    '''
    if return_copy:
        n = deepcopy(n)

    while len(n.internalBuckets) > 0:
    	merge_nodes(n, simple_heuristic, False)

    return n

def merge_bonds(network, node, return_copy):
	'''
	Merges all bonds shared between the specified node and any other node in the network.
	'''
	if return_copy:
		network = deepcopy(network)
		node = list(n for n in network.nodes if n.id == node.id)[0]

	if hasattr(node.tensor, 'network') and len(node.tensor.network.nodes) > mergeCut:
		done = False
		while len(node.tensor.network.nodes) > mergeCut and not done:
			merged = network.mergeClosestLinks(node, compress=True, accuracy=node.tensor.network.accuracy)
			if merged is None:
				done = True
			else:
				node.eliminateLoops()
				merged.eliminateLoops()

	return network, node

