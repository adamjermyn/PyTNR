

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


def merge_rank_2_einsum(args):
    '''
    Recursively merges all rank-2 objects in the specified einsum contraction.
    
    :param args: einsum contraction specification
    :return: argsNew: New einsum contraction specification with no rank-2 objects.
    '''
    
    tensors = args[::2]
    indices = args[1::2]
    out = indices[-1]
    
    # Find rank-2 objects
    rank2 = list([i for i in range(len(tensors)) if len(tensors[i].shape) == 2])
    done = []
    while len(rank2) > 0:
        i = rank2[0]
        if indices[i][0] in out and indices[i][1] in out:
            # Means this is fully external
            done.append(i)
            rank2.pop(0)
    #    else:
            
        
    
