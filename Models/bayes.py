import numpy as np
from math import factorial

from TNR.Network.link import Link
from TNR.Network.node import Node
from TNR.Network.network import Network
from TNR.TreeTensor.identityTensor import IdentityTensor
from TNR.Tensor.arrayTensor import ArrayTensor

def BayesTest1(observations, discreteG, discreteQ, discreteW, discreteH, accuracy):
	'''
	observations is a list of (k,M) pairs
	where k is the number of heads and M-k is the
	number of tails in a repeated Bernoulli coin toss.

	This model represents the likelihood

	L = M! p^k (1-p)^(M-k)/(k!(M-k)!)

	summed over all coins that were observed.

	Here we model

	p_i = min(g*h_i + q^w, 1)

	where each of g, q, w and h_i lie in [0,1]
	and have uniform priors. g, w and q are global parameters.
	
	discrete(G,W,Q) specify the g, w and q values to sample.
	discreteH is the same for h_i.
	'''


	network = Network()

	# Global tensors
	n = len(observations)
	g = Node(IdentityTensor(len(discreteG), n + 1, accuracy=accuracy))
	q = Node(IdentityTensor(len(discreteQ), n + 1, accuracy=accuracy))
	w = Node(IdentityTensor(len(discreteW), n + 1, accuracy=accuracy))

	# Local tensors
	hs = []


	for i,obs in enumerate(observations):
		arr = np.zeros((len(discreteG), len(discreteQ), len(discreteW), len(discreteH)))
		for j,gg in enumerate(discreteG):
			for k,qq in enumerate(discreteQ):
				for e,ww in enumerate(discreteW):
					for l,h in enumerate(discreteH):
						p = min(gg*h + qq**ww, 1)
						arr[j,k,e,l] = factorial(obs[1]) * p**obs[0] * (1 - p)**(obs[1] - obs[0]) / (factorial(obs[0]) * factorial(obs[1] - obs[0]))

		# Marginalizes over all of the individual distributions
		arr = np.sum(arr, axis=-1)
		h = Node(ArrayTensor(arr))
		hs.append(h)
		Link(h.buckets[0], g.buckets[i])
		Link(h.buckets[1], q.buckets[i])
		Link(h.buckets[2], w.buckets[i])

	# Assemble the network
	network.addNode(g)
	network.addNode(q)
	network.addNode(w)
	for h in hs:
		network.addNode(h)

	return network


def BayesTest2(observations, discreteG, discreteQ, discreteW, discreteH, accuracy):
	'''
	observations is a list of (k,M) pairs
	where k is the number of heads and M-k is the
	number of tails in a repeated Bernoulli coin toss.

	This model represents the likelihood

	L = M! p^k (1-p)^(M-k)/(k!(M-k)!)

	summed over all coins that were observed.

	Here we model

	p_i = min(g*h_i + q^w, 1)

	where each of g, q, w and h_i lie in [0,1]
	and have uniform priors. g, w and q are global parameters.
	
	discrete(G,W,Q) specify the g, w and q values to sample.
	discreteH is the same for h_i.
	'''

	network = Network()
 
	# Local tensors
	hs = []

	for i,obs in enumerate(observations):
		arr = np.zeros((len(discreteG), len(discreteQ), len(discreteW), len(discreteH)))
		for j,gg in enumerate(discreteG):
			for k,qq in enumerate(discreteQ):
				for e,ww in enumerate(discreteW):
					for l,h in enumerate(discreteH):
						p = min(gg*h + qq**ww, 1)
						arr[j,k,e,l] = factorial(obs[1]) * p**obs[0] * (1 - p)**(obs[1] - obs[0]) / (factorial(obs[0]) * factorial(obs[1] - obs[0]))

		# Marginalizes over all of the individual distributions
		arr = np.sum(arr, axis=-1)
		h = Node(ArrayTensor(arr))
		hs.append(h)

	extG = [h.buckets[0] for h in hs]
	extW = [h.buckets[1] for h in hs]
	extQ = [h.buckets[2] for h in hs]

	nodes = []

	dimension = len(discreteG)
	while len(extG) > 1:
		iden = np.zeros((dimension,dimension,dimension))
		for i in range(dimension):
			iden[i,i,i] = 1.0
		n = Node(IdentityTensor(dimension, 3, accuracy=accuracy))
		nodes.append(n)
		Link(n.buckets[0], extG[0])
		Link(n.buckets[1], extG[1])
		extG.append(n.buckets[2])
		extG = extG[2:]

	dimension = len(discreteW)
	while len(extW) > 1:
		iden = np.zeros((dimension,dimension,dimension))
		for i in range(dimension):
			iden[i,i,i] = 1.0
		n = Node(IdentityTensor(dimension, 3, accuracy=accuracy))
		nodes.append(n)
		Link(n.buckets[0], extW[0])
		Link(n.buckets[1], extW[1])
		extW.append(n.buckets[2])
		extW = extW[2:]

	dimension = len(discreteQ)
	while len(extQ) > 1:
		iden = np.zeros((dimension,dimension,dimension))
		for i in range(dimension):
			iden[i,i,i] = 1.0
		n = Node(IdentityTensor(dimension, 3, accuracy=accuracy))
		nodes.append(n)
		Link(n.buckets[0], extQ[0])
		Link(n.buckets[1], extQ[1])
		extQ.append(n.buckets[2])
		extQ = extQ[2:]

	for h in hs:
		network.addNode(h)

	for n in nodes:
		network.addNode(n)

	return network
