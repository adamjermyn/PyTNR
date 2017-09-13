from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Network.traceMin import traceMin
import numpy as np
import networkx

def utilHeuristic(n):
	biggest = [-1e100, None, None]

	for n1 in n.nodes:
		for n2 in n1.connectedNodes:
			if n1.tensor.rank <= 2 or n2.tensor.rank <= 2:
				return [1e20, n1, n2]

	for n1 in n.nodes:
		for n2 in n1.connectedNodes:
			length = len(n1.linksConnecting(n2))
			connected = n1.connectedNodes
			connected.update(n2.connectedNodes)
			connected.remove(n1)
			connected.remove(n2)
			t, b = n.dummyMergeNodes(n1, n2)
			tm = traceMin(t.network,None)
			util = (length**2)/(n1.tensor.size*n2.tensor.size)
			util /= (1 + tm.util)**0.5
			if util > biggest[0]:
				biggest = [util, n1, n2]

	return biggest

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
			metric *= 0.7**len(commonNodes)
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

def oneLoopHeuristic(n):
	'''
	This method estimates the contraction in a network which maximizes the size of the loop which
	is eliminated in the process.
	'''
	node=  next(iter(n.nodes))

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
