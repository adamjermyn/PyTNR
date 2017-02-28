from TNRG.TreeTensor.treeTensor import TreeTensor

def entropyHeuristic(n):
	'''
	This method estimates the contraction in a network n which minimizes the resulting network entropy.
	'''
	smallest = [1e20,None,None]
	for nn in n.nodes:
		for nnn in nn.connectedNodes:
			length = nn.linksConnecting(nnn)[0].bucket1.size
			metric = nn.tensor.size*nnn.tensor.size/length**2
			commonNodes = set(nn.connectedNodes).intersection(nnn.connectedNodes)
			metric *= 0.3**len(commonNodes)
			metric = metric - nn.tensor.size - nnn.tensor.size
			if metric < smallest[0]:
				smallest[0] = metric
				smallest[1] = nn
				smallest[2] = nnn
	return smallest

def mergeHeuristic(n):
	'''
	This method estimates the contraction in a network n which maximizes the number of merged links.
	'''
	biggest = [-1e20,None,None]
	for nn in n.nodes:
		for nnn in nn.connectedNodes:
			if nnn.tensor.rank <= 2 or nn.tensor.rank <= 2:
				return [1e20, nn, nnn]
			commonNodes = set(nn.connectedNodes).intersection(nnn.connectedNodes)
			metric = len(commonNodes)
			if metric > biggest[0]:
				biggest = [metric, nn, nnn]
	return biggest

def smallLoopHeuristic(n):
	'''
	This method estimates the contraction in a network which minimizes the size of the loop which
	is eliminated in the process while penalizing rank increases.

	In particular, this heuristic handles all rank <= 2 objects first as well as all objects
	which are not yet tree tensors.

	It then weights subsequent contractions as
		(length of biggest loop) + (tensor 1 rank) + (tensor 2 rank) - (# common nodes)
	and picks the smallest weight. This means that it prioritizes handling smaller tensors,
	handling smaller loops, and handling pairs of tensors which share many nodes.
	'''
	smallest = [1e20, None, None]
	for nn in n.nodes:
		for nnn in nn.connectedNodes:
			indices = nn.indicesConnecting(nnn)
			length = 1
			if not hasattr(nn.tensor, 'network') or not hasattr(nnn.tensor, 'network'):
				length = -100
			elif nn.tensor.rank <= 2 or nnn.tensor.rank <= 2:
				length = -100
			else:
				for i in range(len(indices[0])):
					for j in range(len(indices[0])):
						if i > j:
							length1 = nn.tensor.distBetween(indices[0][i],indices[0][j])
							if length1 > length:
								length = length1
							length1 = nnn.tensor.distBetween(indices[1][i],indices[1][j])
							if length1 > length:
								length = length1
				commonNodes = set(nn.connectedNodes).intersection(nnn.connectedNodes)
				length -= len(commonNodes)
				length += nn.tensor.rank
				length += nnn.tensor.rank
			metric = length
			if metric < smallest[0]:
				smallest[0] = metric
				smallest[1] = nn
				smallest[2] = nnn
	return smallest

def loopHeuristic(n):
	'''
	This method estimates the contraction in a network which maximizes the size of the loop which
	is eliminated in the process.
	'''
	biggest = [-1, None, None]
	for nn in n.nodes:
		for nnn in nn.connectedNodes:
			indices = nn.indicesConnecting(nnn)
			length = 1
			if not hasattr(nn.tensor, 'network') or not hasattr(nnn.tensor, 'network'):
				length = 100
			else:
				for i in range(len(indices[0])):
					for j in range(len(indices[0])):
						if i > j:
							length1 = nn.tensor.distBetween(indices[0][i],indices[0][j])
							if length1 > length:
								length = length1
							length1 = nnn.tensor.distBetween(indices[1][i],indices[1][j])
							if length1 > length:
								length = length1
			if length > biggest[0]:
				biggest[0] = length
				biggest[1] = nn
				biggest[2] = nnn
	return biggest

def oneLoopHeuristic(n, node):
	'''
	This method estimates the contraction in a network which maximizes the size of the loop which
	is eliminated in the process.
	'''
	biggest = [100000, None, None]
	for nnn in node.connectedNodes:
		indices = node.indicesConnecting(nnn)
		length = 10000
		if not hasattr(node.tensor, 'network') or len(indices[0]) == 1:
			length = -100
		else:
			for i in range(len(indices[0])):
				for j in range(len(indices[0])):
					if i > j:
						length1 = node.tensor.distBetween(indices[0][i],indices[0][j])
						if length1 < length:
							length = length1
		if length < biggest[0]:
			biggest[0] = length
			biggest[1] = node
			biggest[2] = nnn
	return biggest

def mergeContractor(n, accuracy, optimize=True, merge=True, mergeCut = 35, verbose=0):
	'''
	This method contracts the network n.

	Optimization and link merging are optional, set by the corresponding named arguments.
	When set to true (default) they are done at each stage, with optimization following merging.

	An entropy heuristic is used to decide the contraction order.

	The named argument verbose controls how much output to print:
		0 - None
		1 - Summary
		2 - Node enumeration
	'''
#	node = next(iter(n.nodes))
	while len(n.nodes) > 1:
#		q, n1, n2 = smallLoopHeuristic(n)
		q, n1, n2 = entropyHeuristic(n)
#		q, n1, n2 = oneLoopHeuristic(n, node)
#		q, n1, n2 = mergeHeuristic(n)

		n3 = n.mergeNodes(n1, n2)

		n3.tensor.eliminateLoops()

		if merge:
			print('MERGE')
			if hasattr(n3.tensor, 'compressedSize') and len(n3.tensor.network.nodes) > mergeCut:
				while len(n3.tensor.network.nodes) > mergeCut:
					merged = n.mergeClosestLinks(n3, compress=True, accuracy=accuracy)
					n3.tensor.eliminateLoops()
					merged.tensor.eliminateLoops()
					if optimize:
						n3.tensor.optimize(verbose=verbose)
						merged.tensor.optimize(verbose=verbose)

			print('MERGEDONE')


		if verbose >= 2:
			for nn in n.nodes:
				if hasattr(nn.tensor,'compressedSize'):
					print(nn.id, nn.tensor.shape, nn.tensor.compressedSize, 1.0*nn.tensor.compressedSize/nn.tensor.size,len(nn.tensor.network.nodes),[qq.tensor.shape for qq in nn.tensor.network.nodes])
#					print(nn.tensor.network)
				else:
					print(nn.tensor.shape, nn.tensor.size)

		if verbose >= 1:
			counter = 0
			for nn in n.nodes:
				if hasattr(nn.tensor, 'network'):
					counter += 1
			print('-------',len(n3.connectedNodes),q,counter,len(n.nodes),'-------')
	return n





