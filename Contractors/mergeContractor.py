from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Network.traceMin import traceMin
import numpy as np
import networkx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['mergeContractor'])

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (config.mem_limit, hard))

def mergeContractor(n, accuracy, heuristic, optimize=True, merge=True, plot=False, mergeCut = 35):
	'''
	This method contracts the network n to the specified accuracy using the specified heuristic.

	Optimization and link merging are optional, set by the corresponding named arguments.
	When set to true (default) they are done at each stage, with optimization following merging.

	The plot option, if True, plots the entire network at each step and saves the result to a PNG
	file in the top-level Overview folder. This defaults to False.
	'''

	if plot:
		pos = None
		counter = 0

	while len(n.nodes) > 1:

		if plot:
			g = n.toGraph()
			reusePos = {}
			if pos is not None:
				for nn in g.nodes():
					if nn in pos:
						reusePos[nn] = pos[nn]
				pos=networkx.fruchterman_reingold_layout(g, pos=reusePos, fixed=reusePos.keys())
			else:
				pos=networkx.fruchterman_reingold_layout(g)
			plt.figure(figsize=(11,11))
			networkx.draw(g, pos, width=2)
			plt.savefig('Overview/'+str(counter) + '.png')
			plt.clf()
			plt.close()
			counter += 1

		q, n1, n2 = heuristic(n)

		n3 = n.mergeNodes(n1, n2)

		n3.eliminateLoops()

		if optimize:
			n3.tensor.optimize()

		if merge:
			logger.info('Merging nodes...')
			nn = n3
			if hasattr(nn.tensor, 'compressedSize') and len(nn.tensor.network.nodes) > mergeCut:
				done = False
				while len(nn.tensor.network.nodes) > mergeCut and not done:
					merged = n.mergeClosestLinks(n3, compress=True, accuracy=accuracy)
					if merged is not None:
						nn.eliminateLoops()
						merged.eliminateLoops()
						if optimize:
							nn.tensor.optimize()
							merged.tensor.optimize()
					else:
						done = True

			logger.info('Merging complete.')

		for nn in n.nodes:
			if hasattr(nn.tensor,'compressedSize'):
				logger.debug(nn.id, nn.tensor.shape, nn.tensor.compressedSize, 1.0*nn.tensor.compressedSize/nn.tensor.size,len(nn.tensor.network.nodes),[qq.tensor.shape for qq in nn.tensor.network.nodes])
			else:
				logger.debug(nn.tensor.shape, nn.tensor.size)

		counter = 0
		for nn in n.nodes:
			if hasattr(nn.tensor, 'network'):
				counter += 1
		logger.info('Network has '+str(len(n.nodes)) + ' nodes, of which ' + str(counter) + ' contain treeTensor objects.')
		logger.info('Contraction had utility ' + str(q) + ' and resulted in a tensor connected to ' + str(len(n3.connectedNodes)) + ' nodes.')


	return n





