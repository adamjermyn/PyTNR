def mergeContractor(n, accuracy, verbose=0):
	'''
	This method contracts the network n with optimization and link merger at each stage.
	An entropy heuristic is used to decide the contraction order.
	The named argument verbose controls how much output to print:
		0 - None
		1 - Summary
		2 - Node enumeration
	'''
	while len(n.nodes) > 1:
		smallest = [1e20,None,None]
		for nn in n.nodes:
			for nnn in nn.connectedNodes:
				length = nn.linksConnecting(nnn)[0].bucket1.size
				metric = nn.tensor.size*nnn.tensor.size/length**2
				commonNodes = set(nn.connectedNodes).intersection(nnn.connectedNodes)
				metric *= 0.75**len(commonNodes)
				metric = metric - nn.tensor.size - nnn.tensor.size
				if metric < smallest[0]:
					smallest[0] = metric
					smallest[1] = nn
					smallest[2] = nnn
		n1 = smallest[1]
		n2 = smallest[2]
		n3 = n.mergeNodes(n1, n2)
		n.mergeLinks(n3, accuracy=accuracy)
		n3.tensor.optimize()

		if verbose >= 2:
			for nn in n.nodes:
				print(nn.tensor.shape, nn.tensor.compressedSize, 1.0*nn.tensor.compressedSize/nn.tensor.size)

		if verbose >= 1:
			print('-------',len(n3.connectedNodes),len(n.nodes),'-------')


	return n