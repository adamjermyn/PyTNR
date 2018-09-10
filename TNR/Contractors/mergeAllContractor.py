from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.Network.traceMin import traceMin
import numpy as np
import networkx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['mergeContractor'])

#import resource
#soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (config.mem_limit, hard))


def mergeContractor(
        n,
        accuracy,
        heuristic,
        optimize=True):
    '''
    This method contracts the network n to the specified accuracy using the specified heuristic.

    Optimization is optional, set by the corresponding named arguments.
    When set to true (default) it is done at each stage.
    '''


    while len(n.internalBuckets) > 0:

        q, n1, n2 = heuristic(n)

        n3 = n.mergeNodes(n1, n2)


    for n1 in n.nodes:
        n1.eliminateLoops()

    return n
