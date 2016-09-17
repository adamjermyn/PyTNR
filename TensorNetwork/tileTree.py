from networkTree import NetworkTree
from latticeNode import latticeNode
from node import Node
from utils import flatten, multiMod
import numpy as np
from copy import deepcopy
from bucket import Bucket
from tensor import Tensor

def makeTileTreeFromArrays(latticeLength, dimensions, bondArrays, footprints):
	nt = NetworkTree()

	siteIndices = np.meshgrid(*list(list(range(d)) for d in dimensions),indexing='ij')
	siteIndices = np.array(siteIndices)
	siteIndices = np.reshape(siteIndices, (len(dimensions),-1))
	siteIndices = np.transpose(siteIndices)

	sites = np.empty(shape=dimensions, dtype='object')

	bonds = [np.empty(shape=dimensions, dtype='object') for _ in bondArrays]

	periodicLinks = [set() for _ in dimensions]

	for si in siteIndices:
		sites[tuple(si)] = latticeNode(latticeLength, nt)
		for i in range(len(bondArrays)):
			bt = Tensor(bondArrays[i].shape, bondArrays[i])
			bonds[i][tuple(si)] = nt.addNodeFromTensor(bt)

	for si in siteIndices:
		for i in range(len(bondArrays)):
			for q,dj in enumerate(footprints[i]):
				x, dims = multiMod(np.array(si) + np.array(dj), dimensions)
				l = sites[tuple(si)].addLink(bonds[i][tuple(x)],q)
				if len(dims) > 0:
					l.setPeriodic()
				for j in dims:
					periodicLinks[j].add(l)

	return nt, periodicLinks

def linkTrees(dimension, tree, periodicLinks):
	nt = NetworkTree()

	periodicLinkID = {l.id():l for l in periodicLinks[dimension]}

	tNew = deepcopy(tree)

	linkPairs = set()

	for l in tNew.allLinks():
		if l.id() in periodicLinkID.keys():
			l1 = periodicLinkID[l.id()]
			l2 = l

			n1 = l1.bucket1().topNode()
			n2 = l2.bucket1().topNode()

			if n1.id() == n2.id():
				b1 = l1.bucket1()
				b2 = l2.bucket1()

				l1.setBucket1(b2)
				l2.setBucket2(b1)
			else:
				b1 = l1.bucket1()
				b2 = l2.bucket2()

				l1.setBucket1(b2)
				l2.setBucket2(b1)

	nt.addNetworkTree(tree)

	newPeriodicLinks = [p for p in periodicLinks]
	newPeriodicLinks[dimension] = set(p[1] for p in linkPairs)

	for p in linkPairs:
		p[0].setNotPeriodic()

	return nt, newPeriodicLinks

