from copy import deepcopy
from TNR import config
import sys


from TNR.Utilities.logger import makeLogger
logger = makeLogger(__name__, config.levels['contractor'])

class contractor:

	def __init__(self, network):
		self.network = deepcopy(network)

	def perform_action(self, action, *args):
		return action(self.network, *args)

class replicaContractor:

	def __init__(self, network, num_copies, cost_cap):
		self.replicas = list(contractor(network) for _ in range(num_copies))
		self.costs = list(c.network.compressedSize for c in self.replicas)
		self.num_copies = num_copies
		self.cost_cap = cost_cap

	def index_of_least_cost(self):
		return 	self.costs.index(min(self.costs))

	def perform_action(self, index, action, *args):
		try:
			ret = self.replicas[index].perform_action(action, *args)
		except KeyboardInterrupt:
			exit()
		except:
			e = sys.exc_info()[0]
			logger.info(str(e))
			logger.info('Error in taking action on network ' + str(ind) + '.')
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

		return ret, replaced
