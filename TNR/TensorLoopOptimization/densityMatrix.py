import numpy as np

from copy import deepcopy
from scipy.linalg import svd

def cost(rank):
    ind0 = list(rank).index(0)
    rank = np.roll(rank, len(rank) - ind0) # Push the zero to the front
    rank = rank[1:] # Cut off the zero
    return np.sum(rank[1:] * rank[:-1])

def cutSVD(loop, environment, tolerance, bids, otherBids):
    '''
    Computes the non-zero singular values of each bond in each possible loop cut.

    :param loop:
    :param environment:
    :param bids:
    :param otherBids:
    :return:
    '''

    ### First contract the loop with the environment.
    # We avoid contracting any links because we need the links in the loop to remain identifiable.
    
    # Identify indexing
    # bids are in order of appearance in loop.externalBuckets
    # otherBids are ordered to correspond to bids
    
    inds = list(range(loop.rank))
    envInds = []
    envBids = list(b.id for b in environment.externalBuckets)
    for bid in otherBids:
        envInds.append(envBids.index(bid))

    # Contract
    net = loop.contract(inds, environment, envInds, elimLoops=False)
    
    ### Next find all internal inks within the result.
    # Because the environment is formed of disjoint pieces, one of which connects to
    # each loop tensor, these are just the links in the original loop.
    lids = list(set(b.link.id for n in loop.network.nodes for b in n.buckets if b.linked))
    links = []
    for n in net.network.nodes:
        for b in n.buckets:
            if b.linked and b.link.id in lids:
                links.append(b.link)
    links = list(set(links))
    lids = list(l.id for l in links)
                
    ### Identify necessary ranks for all cuts

    ranks = np.zeros((len(links), len(links)))
    
    # Iterate over pairs of links
    for i in range(1,len(links)):
        for j in range(i):
            l1 = links[i]
            l2 = links[j]
                        
            mat = prepareTensors(net, l1, l2)
            s = svd(mat, compute_uv=False)
            p = s / np.sum(s) # We are already working with a density matrix, so no need to square the eigenvalues.
            cp = 1 - np.cumsum(p)

            # We divide the tolerance by the rank so the accumulated L2 error is below the threshold.
            ind = np.searchsorted(cp[::-1], tolerance / loop.rank, side='left') 
            ind -= 1 # Because it searches until it hits something bigger than tolerance
            ind = len(cp) - ind
            print(p, ind)

            
            ranks[i,j] = ind
            ranks[j,i] = ranks[i,j]

    return ranks, list(cost(r) for r in ranks), lids


def prepareTensors(net, link1, link2):
    # Copy the NetworkTensor
    net = deepcopy(net)

    # Record bucket ID's:
    bid11 = link1.bucket1.id
    bid12 = link1.bucket2.id
    bid21 = link2.bucket1.id
    bid22 = link2.bucket2.id
    
    # Identify links in the copied NetworkTensor
    for n in net.network.nodes:
        for b in n.buckets:
            if b.id == bid11 or b.id == bid12:
                link1 = b.link
            elif b.id == bid21 or b.id == bid22:
                link2 = b.link

    # Delete specified links from loop
    assert link1.bucket1 in net.network.internalBuckets
    assert link1.bucket2 in net.network.internalBuckets
    assert link2.bucket1 in net.network.internalBuckets
    assert link2.bucket2 in net.network.internalBuckets
    net.network.removeLink(link1)
    net.network.removeLink(link2)
    net.externalBuckets.append(link1.bucket1)
    net.externalBuckets.append(link1.bucket2)
    net.externalBuckets.append(link2.bucket1)
    net.externalBuckets.append(link2.bucket2)

    # The final four indices on net now correspond to link1.bucket1, link1.bucket2, link2.bucket1
    # and link2.bucket2 in that order.

    # Copy the network and grab its bucket indices.
    cnet = net.copy()
    bids = list(b.id for b in net.externalBuckets)
    newBids = list(b.id for b in cnet.externalBuckets)
    bdict = {bid:nbid for bid,nbid in zip(*(bids, newBids))}

    # Contract network against itself
    # We omit the final four indices from the contraction because those are the ones we just formed
    # by cutting link1 and link2.
    net = net.contract(range(net.rank - 4), cnet, range(net.rank - 4), elimLoops=False)

    # There are now two disjoint components in net. Each has two external buckets corresponding to
    # each of the two copies of link1 and link2 in the network. We read these tensors out as arrays.
    # By construction their indices already correspond to one another
    # (e.g. first contracts with first, second with second, etc.).

    # We don't need the overall scale so we discard the logarithmic part.
    arrs, buckets, _ = net.disjointArrays    
    
    # Now we transpose these arrays so that the first two indices contain one bucket from each of
    # link1 and link2.
    
    link1ids = set([bid11, bid12, bdict[bid11], bdict[bid12]])
    link2ids = set([bid21, bid22, bdict[bid21], bdict[bid22]])
    
    if (buckets[0][0] in link1ids and buckets[0][1] in link1ids) or (buckets[0][0] in link2ids and buckets[0][1] in link2ids):
        # Means we need to swap two indices. We arbitrarily choose the middle two.
    
        arrs[0] = np.transpose(arrs[0], axes=[0, 2, 1, 3])
        arrs[1] = np.transpose(arrs[1], axes=[0, 2, 1, 3])

    arrs[0] = np.reshape(arrs[0], (arrs[0].shape[0]*arrs[0].shape[1], arrs[0].shape[2]*arrs[0].shape[3]))
    arrs[1] = np.reshape(arrs[1], (arrs[1].shape[0]*arrs[1].shape[1], arrs[1].shape[2]*arrs[1].shape[3]))

    # Now we pull arrs[0] apart using SVD
    u, s, v = svd(arrs[0], full_matrices=False)
    u = np.dot(u, np.diag(np.sqrt(s)))
    v = np.dot(np.diag(np.sqrt(s)), v)
    
    
    # Finally we multiply v . arrs[1] . u
    ret = np.einsum('ij,jk,kl->il', v, arrs[1], u)

    return ret    

def densityMatrix(loop, environment, index1, index2):
    '''
    Arguments:
        loop        -   A NetworkTensor containing a loop.
        environment -   A NetworkTensor containing the environment of the loop.
                        This should have no cycles.
        index1      -   The index to the left of the first bond of interest.
        index2      -   The index to the left of the second bond of interest.

    Let A be the tensor given by the portion of the loop between index1 (exclusive) and
    index2 (inclusive), and let C be the complement. Let envA and envC be the portions
    of the environment tensor which contract against A and C respectively.

    The reduced density matrix after tracing out the external legs of C is

    R = envA . A . C . envC . envC* . C* . A* . envA*

    where the dot product means summation over shared indices and * denotes a transpose.
    Note that the structure of this network is such that the above notation is associative.

    Let

    D = C . envC . envC* . C*

    This is Hermitian, so Sqrt[D] exists. Hence with

    K = envA . A . Sqrt[D]

    we find

    R = K . K*.

    The conjugate density matrix is given by

    Q = K* . K = Sqrt[D]* . envA* . A* . A . envA . Sqrt[D]

    This method returns Q.

    Note:

    It can be shown that R and Q have the same non-zero eigenvalues. This is because the
    non-zero eigenvalues of AB are the same as those of BA for any A and B, which in turn
    follows from the fact that the coefficients of the characteristic polynomial of a matrix
    M depend only on powers of trace(M^n) and trace((AB)^n) = trace((BA)^n).

    This is useful because Q is much smaller in size than R and so can be used to efficiently
    compute the bond dimensions of cuts.
    '''