from copy import deepcopy

from TNR.TensorLoopOptimization.optimizer import optimize as opt
from TNR.Environment.environment import artificialCut, identityEnvironment, fullEnvironment

def loop_svd_elim(tensor, return_copy):
    if return_copy:
        tensor = deepcopy(tensor)

    canon = lambda x: list(y for y in tensor.network.nodes for i in range(len(x)) if y.id == x[i])
    prodkey = lambda x: sum(x[i].tensor.size*x[i+1].tensor.size for i in range(len(x)-1))
    while len(networkx.cycles.cycle_basis(tensor.network.toGraph())) > 0:

        tensor.contractRank2()
        cycles = sorted(networkx.cycles.cycle_basis(tensor.network.toGraph()), key=len)
        if len(cycles) > 0:
            print('Cycles:',len(cycles), list(len(c) for c in cycles))
            old_nodes = set(tensor.network.nodes)

            tensor.cutLoop(cycles[0], False)
            tensor.contractRank2()
            new_nodes = set(tensor.network.nodes)

            affected = set(cycles[0])
            affected.update(new_nodes.difference(old_nodes))
            tensor.network.graph = None

    assert len(networkx.cycles.cycle_basis(tensor.network.toGraph())) == 0

    return tensor
    
def cutLoop(tensor, loop, return_copy, cutIndex=None):
    logger.debug('Cutting loop.')
    print(len(loop))
    tensor.network.check()

    # Form the environment network

#        environment, net, internalBids, envBids = artificialCut(tensor, loop)
    environment, net, internalBids, envBids = identityEnvironment(tensor, loop)
    bids = list([b.id for b in net.externalBuckets])

    # Determine optimal cut
    ranks, costs, lids = cutSVD(net, environment, tensor.accuracy, bids, envBids)

    # Assocaite links with ranks
    links = set()
    for n in net.network.nodes:
        for b in n.buckets:
            if b.linked:
                links.add(b.link)

    freeBucket = lambda x: list(b for b in x.buckets if not b.linked)[0]
    rankDict = {}
    for i,l in enumerate(links):
        for j,l2 in enumerate(links):
            if i != j:
                n11 = l.bucket1.node
                n12 = l.bucket2.node
                n21 = l2.bucket1.node
                n22 = l2.bucket2.node


                leftBids = set()
                current = n11
                prev = n12
                while True:
                    leftBids.add(freeBucket(current).id)
                    if current == n21 or current == n22:
                        break
                    found = False
                    for n in current.connectedNodes:
                        if n != prev:
                            prev = current
                            current = n
                            found = True
                            break

                    if not found:
                        break

                rightBids = set(b.id for b in net.externalBuckets).difference(leftBids)

                leftBids = frozenset(leftBids)
                rightBids = frozenset(rightBids)

                rankDict[(leftBids, rightBids)] = ranks[lids.index(l.id), lids.index(l2.id)]
                rankDict[(rightBids, leftBids)] = ranks[lids.index(l.id), lids.index(l2.id)]


    ind = np.argmin(costs)
    
    ranks = ranks[ind]
    ranks[ranks == 0] = 1
    
    logger.debug('Final ranks: ' + str(ranks))

    # Cut
    for i,r in enumerate(ranks):
        if r == 1:
            lid = lids[i]
    for n in net.network.nodes:
        for b in n.buckets:
            if b.linked and b.link.id == lid:
                l = b.link

    newNet = svdCut(net, environment, l, bids, envBids, rankDict)

    ranks2 = []
    doneLinks = set()
    for n in newNet.network.nodes:
        for b in n.buckets:
            if b.linked and b.link not in doneLinks:
                doneLinks.add(b.link)
                ranks2.append(b.size)

    logger.debug('actual ranks: ' + str(ranks2) + ', predicted: ' + str(ranks))

    # Now put the new nodes in the network and replace the loop
    netBids = list(b.id for b in net.externalBuckets)

    toRemove = []
    removedBuckets = []
    for n in tensor.network.nodes:
        nbids = set(b.id for b in n.buckets)
        if len(nbids.intersection(netBids)) > 0:
            toRemove.append(n)
    for n in toRemove:
        tensor.network.removeNode(n)
        removedBuckets.extend(n.buckets)
                    
    existingBuckets = {}
    for b in removedBuckets:
        existingBuckets[b.id] = b
    
    newNodes = list(newNet.network.nodes)
    for n in newNodes:
        newNet.network.removeNode(n)
        
        newBuckets = []
        for b in n.buckets:
            if b.id in netBids:
                oldB = existingBuckets[b.id]
                newBuckets.append(oldB)
            else:
                newBuckets.append(b)

        n.buckets = newBuckets
        for b in n.buckets:
            b.node = n

        tensor.network.addNode(n)
    
    tensor.network.cutLinks()
    tensor.network.check()

    logger.debug('Cut.')
    return tensor
