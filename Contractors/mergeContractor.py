from TNRG.TreeTensor.treeTensor import TreeTensor
from TNRG.Network.traceMin import traceMin
import numpy as np
import networkx
import matplotlib.pyplot as plt
import matplotlib.cm as cm




def mergeContractor(n, accuracy, heuristic, optimize=True, merge=True, plot=False, mergeCut = 35):
	'''
	This method contracts the network n to the specified accuracy using the specified heuristic.

	Optimization and link merging are optional, set by the corresponding named arguments.
	When set to true (default) they are done at each stage, with optimization following merging.

	The plot option, if True, plots the entire network at each step and saves the result to a PNG
	file in the top-level Overview folder. This defaults to False.

	The named argument verbose controls how much output to print:
		0 - None
		1 - Summary
		2 - Node enumeration
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
			n3.tensor.optimize(verbose=verbose)

		if merge:
			print('MERGE')
			nn = n3
			if hasattr(nn.tensor, 'compressedSize') and len(nn.tensor.network.nodes) > mergeCut:
				done = False
				while len(nn.tensor.network.nodes) > mergeCut and not done:
					merged = n.mergeClosestLinks(n3, compress=True, accuracy=accuracy)
					if merged is not None:
						nn.eliminateLoops()
						merged.eliminateLoops()
						if optimize:
							nn.tensor.optimize(verbose=verbose)
							merged.tensor.optimize(verbose=verbose)
					else:
						done = True

			print('MERGEDONE')


		if verbose >= 2:
			for nn in n.nodes:
				if hasattr(nn.tensor,'compressedSize'):
					print(nn.id, nn.tensor.shape, nn.tensor.compressedSize, 1.0*nn.tensor.compressedSize/nn.tensor.size,len(nn.tensor.network.nodes),[qq.tensor.shape for qq in nn.tensor.network.nodes])
				else:
					print(nn.tensor.shape, nn.tensor.size)

		if verbose >= 1:
			counter = 0
			for nn in n.nodes:
				if hasattr(nn.tensor, 'network'):
					counter += 1
			print('-------',len(n3.connectedNodes),q,counter,len(n.nodes),'-------')


	return n





