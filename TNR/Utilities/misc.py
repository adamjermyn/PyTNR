import networkx
from copy import deepcopy

################################
# Miscillaneous Helper Functions
################################

def tupleReplace(tpl, i, j):
    '''
    Returns a tuple with element i of tpl replaced with the quantity j.
    If j is None, just removes element i.
    '''
    assert i >= 0
    assert i < len(tpl)

    tpl = list(tpl)
    if j is not None:
        tpl = tpl[:i] + [j] + tpl[i + 1:]
    else:
        tpl = tpl[:i] + tpl[i + 1:]
    return tuple(tpl)

def cutSlice(i, shape):
    '''
    Creates a slice object which selects the zero position on index i for
    an array of the specified shape.
    :param i: The index to slice.
    :param shape: The shape of the array.
    :return: sl: A slice object.
    '''

    sl = list(slice(0, shape[k]) for k in range(i)) + [0] + list(
        slice(0, shape[k]) for k in range(i + 1, len(shape)))
    return sl

def shortest_cycles(graph):
    graph = networkx.relabel_nodes(graph, lambda x: x.id)
    cycles = []
    for e in graph.edges():
        n1 = e[0]
        n2 = e[1]
        g = deepcopy(graph)
        g.remove_edge(n1, n2)
        try:
            path = networkx.shortest_path(g, n1,  n2)
            cycles.append(path)
        except networkx.exception.NetworkXNoPath:
            break
    return cycles


def nodes_to_einsum(nodes):
    '''
    Produces a list of arrays and indices to pass to einsum based
    on the specified set of nodes. Only nodes within this set will
    be considered for contraction.
    
    :param nodes: Set of nodes to be contracted.
    :return: args: Arguments to pass to einsum.
    :return: bids: ID's of external buckets of the contraction, ordered in the same manner as they will appear in the contracted array.
    '''

    # Order nodes
    nodeList = list(nodes)

    # Accumulate tensors
    tensors = list(n.tensor.scaledArray for n in nodeList)

    # Setup subscript arrays
    subs = list([-1 for _ in range(len(n.buckets))] for n in nodeList)
    out = []
    bids = []

    # Construct lists
    counter = 0
    for i in range(len(nodeList)):
        n = nodeList[i]

        for j,b in enumerate(n.buckets):
            if subs[i][j] == -1:
                subs[i][j] = counter
                
                if b.linked and b.otherNode in nodes:
                    ind = nodeList.index(b.otherNode)
                    ind2 = nodeList[ind].buckets.index(b.otherBucket)
                    if b.size == 1:
                        # Need to cut this link
                        subs[i][j] = -2
                        subs[ind][ind2] = -2
                        counter -= 1
                    else:
                        subs[ind][ind2] = counter
                else:
                    out.append(counter)
                    bids.append(b.id)
                counter += 1

    # Cut marked links
    for i in range(len(subs)):
        j = 0
        while j < len(subs[i]):
            if subs[i][j] == -2:
                tensors[i] = tensors[i][cutSlice(j, tensors[i].shape)]
                subs[i].pop(j)
            else:
                j += 1
        
    args = []
    for i in range(len(nodeList)):
        args.append(tensors[i])
        args.append(subs[i])
    args.append(out)
    
    return args, bids


