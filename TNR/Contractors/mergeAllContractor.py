from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['mergeContractor'])

#import resource
#soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (config.mem_limit, hard))


def mergeContractor(
        n,
        accuracy,
        optimize=True):
    '''
    This method contracts the network n to the specified accuracy using the specified heuristic.

    Optimization is optional, set by the corresponding named arguments.
    When set to true (default) it is done at each stage.
    '''


    while len(n.internalBuckets) > 0:

        n1 = None
        while n1 is None:
            n1 = next(iter(n.nodes))
            if len(n1.connectedNodes) == 0:
                n1 = None
        
        n2 = next(iter(n1.connectedNodes))

        n3 = n.mergeNodes(n1, n2)


    for n1 in n.nodes:
        n1.eliminateLoops()

    return n
