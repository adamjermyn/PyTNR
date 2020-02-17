from copy import deepcopy

def optimize(tensor, return_copy):
    '''
    Optimizes the tensor network to minimize memory usage.
    '''

    if return_copy:
        tensor = deepcopy(tensor)

    logger.info('Optimizing tensor with shape ' + str(tensor.shape) + '.')

    s2 = 0
    for n in tensor.network.nodes:
        s2 += n.tensor.size

    logger.info('Stage 1: Contracting Rank-2 Tensors and Double Links.')
    tensor.contractRank2()

    logger.info('Stage 2: Optimizing Links.')

    while len(
        tensor.optimized.intersection(
            tensor.network.internalBuckets)) < len(
            tensor.network.internalBuckets):
        b1 = next(
            iter(
                tensor.network.internalBuckets.difference(
                    tensor.optimized)))
        b2 = b1.otherBucket
        n1 = b1.node
        n2 = b2.node

        if n1.tensor.rank < 3 or n2.tensor.rank < 3:
            tensor.optimized.add(b1)
            tensor.optimized.add(b2)
            continue

        sh1 = n1.tensor.shape
        sh2 = n2.tensor.shape
        s = n1.tensor.size + n2.tensor.size

        logger.debug('Optimizing tensors ' +
                     str(n1.id) +
                     ',' +
                     str(n2.id) +
                     ' with shapes' +
                     str(n1.tensor.shape) +
                     ',' +
                     str(n2.tensor.shape))

        t, buckets = tensor.network.dummyMergeNodes(n1, n2)
        arr = t.array
        if n1.tensor.rank == 3:
            ss = set([0, 1])
        elif n2.tensor.rank == 3:
            ss = set([2, 3])
        else:
            ss = None

        logger.debug('Computing minimum entropy cut...')
        best = entropy(arr, pref=ss)
        logger.debug('Done.')

        if set(best) != ss and set(best) != set(
                range(n1.tensor.rank + n2.tensor.rank - 2)).difference(ss):
            n = tensor.network.mergeNodes(n1, n2)
            nodes = tensor.network.splitNode(n, ignore=best)

            for b in nodes[0].buckets:
                tensor.optimized.discard(b)
            for b in nodes[1].buckets:
                tensor.optimized.discard(b)

            if nodes[0] in nodes[1].connectedNodes:
                assert len(nodes) == 2
                l = nodes[0].findLink(nodes[1])
                tensor.optimized.add(l.bucket1)
                tensor.optimized.add(l.bucket2)
                logger.debug('Optimizer improved to shapes to' +
                             str(nodes[0].tensor.shape) +
                             ',' +
                             str(nodes[1].tensor.shape))
            else:
                # This means the link was cut
                logger.debug('Optimizer cut a link. The resulting shapes are ' +
                             str(nodes[0].tensor.shape) + ', ' + str(nodes[1].tensor.shape))

        else:
            tensor.optimized.add(b1)
            tensor.optimized.add(b2)

        logger.debug('Optimization steps left:' + str(-len(tensor.optimized.intersection(
            tensor.network.internalBuckets)) + len(tensor.network.internalBuckets)))

    s1 = 0
    for n in tensor.network.nodes:
        s1 += n.tensor.size
    logger.info('Optimized network with shape ' +
                str(tensor.shape) +
                ' and ' +
                str(len(tensor.network.nodes)) +
                ' nodes. Size reduced from ' +
                str(s2) +
                ' to ' +
                str(s1) +
                '.')

    return tensor