from TNR.Utilities.logger import makeLogger
from TNR.Contractors.contractor import contractor
from TNR.Contractors.heuristics import simple_heuristic as heuristic

def mergeAllContractor(n):
    '''
    This method contracts the network n to the specified accuracy using the specified heuristic.

    Optimization is optional, set by the corresponding named arguments.
    When set to true (default) it is done at each stage.
    '''

    c = contractor(n)
    done = False
    while not done:
        node, done = c.take_step(heuristic, eliminateLoops=False)
        if optimize:
            c.optimize(new_node)
    n = c.network

    for node in n.nodes:
        node.eliminateLoops()

    return n
