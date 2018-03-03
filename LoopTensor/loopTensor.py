from operator import mul
from copy import deepcopy
from collections import defaultdict

import itertools as it
import numpy as np
import operator

from TNR.Tensor.tensor import Tensor
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Network.treeNetwork import TreeNetwork
from TNR.Network.node import Node
from TNR.Network.link import Link
from TNR.Network.bucket import Bucket
from TNR.Utilities.svd import entropy
from TNR.TensorLoopOptimization.loopOpt import optimizeNorm as optimize
from TNR.TensorLoopOptimization.optimizer import cut

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])

class LoopTensor(Tensor):

	def __init__(self, network, loopNodes, accuracy):
		self.accuracy = accuracy
		self.network = deepcopy(network)

		# Prune the network down to just the loop
		ids = list([n.id for n in loopNodes])
		for n in self.network.nodes:
			if n.id not in ids:
				self.network.removeNode(n)

	### TODO: Finish implementing this