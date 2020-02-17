from copy import deepcopy
from TNR.Utilities.logger import makeLogger
from TNR import config
import sys
logger = makeLogger(__name__, config.levels['contractor'])

class contractor:

	def __init__(self, network):
		self.network = deepcopy(network)

	def take_step(self, heuristic, eliminateLoops=True):
		'''
		Uses a heuristic to pick the next contraction and performs it.

		If the result is a tree tensor and the user specifies, this routine then eliminates any resulting loops.
		'''

		# Pre-contraction output
		logger.info('Network has ' +
					str(len(self.network.nodes)) +
					' nodes.')

		# Contraction
		utility, node1, node2 = heuristic(self.network)
		new_node = self.network.mergeNodes(node1, node2)

		# Post-contraction output
		logger.info('Contraction had utility ' +
					str(utility) +
					' and resulted in a tensor connected to ' +
					str(len(new_node.connectedNodes)) +
					' nodes.')

		if eliminateLoops:
			new_node.eliminateLoops()

		done = False
		if len(self.network.internalBuckets) == 0:
			done = True

		return new_node, done

	def merge_bonds(self, node):
		'''
		Merges all bonds shared between the specified node and any other node in the network.
		'''
		if hasattr(node.tensor, 'network') and len(node.tensor.network.nodes) > mergeCut:
			done = False
			while len(node.tensor.network.nodes) > mergeCut and not done:
				merged = self.network.mergeClosestLinks(node, compress=True, accuracy=node.tensor.network.accuracy)
				if merged is None:
					done = True
				else:
					node.eliminateLoops()
					merged.eliminateLoops()

	def optimize(self, node):
		'''
		Optimizes the tree tensor of the specified node.
		'''
		node.tensor.optimize()

class replicaContractor(contractor):

	def __init__(self, network, num_copies, cost_cap):
		self.replicas = list(contractor(network) for _ in range(num_copies))
		self.costs = list(c.network.compressedSize for c in self.replicas)
		self.num_copies = num_copies
		self.cost_cap = cost_cap

	def take_step(self, heuristic, eliminateLoops=True):
		replaced = False

		# Work with the lowest-cost network
		ind = self.costs.index(min(self.costs))
		c = self.replicas[ind]

		# Perform one contraction step
		new_node = None
		done = None
		new_node, done = c.take_step(heuristic, eliminateLoops=eliminateLoops)
		if False:
			try:
				new_node, done = c.take_step(heuristic, eliminateLoops=eliminateLoops)
			except KeyboardInterrupt:
				exit()
			except:
				e = sys.exc_info()[0]
				logger.info(str(e))
				logger.info('Failed to contract network ' + str(ind) + '.')
				logger.info('Replacing that with a clone of the next best network.')
				# Clone the current best network in place of the failed one
				del self.replicas[ind]
				del self.costs[ind]
				ind = self.costs.index(min(self.costs))
				self.replicas.append(deepcopy(self.replicas[ind]))
				self.costs.append(self.costs[ind])
				replaced = True

		# Check if cost cap was exceeded. If so, remove the offending replica and replace it with a copy of the cheapest one.
		if self.cost_cap is not None:
			for i in range(self.num_copies):
				if self.costs[i] > self.cost_cap:
					logger.info('Network ' + str(i) + ' has exceeded the cost cap. Replacing it with a clone of the best network.')
					self.costs[i] = min(costs)
					self.replicas[i] = deepcopy(self.replicas[self.costs.index(min(self.costs))])
				self.replicas.append(deepcopy(self.replicas[ind]))
				self.costs.append(self.costs[ind])
				replaced = True

		return new_node, done, ind, replaced

	def merge_bonds(self, node, ind):
		self.replicas[ind].merge_bonds(node)

	def optimize(self, node, ind):
		self.replicas[ind].optimize(node)