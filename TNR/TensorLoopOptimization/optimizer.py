import numpy as np
from copy import deepcopy

from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TensorLoopOptimization.optTensor import optTensor

from TNR.Utilities.graphPlotter import plot

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])


'''

Note that it is not faster to optimize by SVD: this will not catch the loop entropy.

Further note that there is a problem with cutting by SVD: It will initially significantly
overestimate the memory cost of the cut because the tensor product will produce the largest
bonds which could possibly be required, and only after O(N) svd operations will these be
significantly reduced (e.g. once the changes propagate along the identity line to the other side).

A possible compromise is therefore to use the least squares solver in the expanding phase
and SVD in the reduction phase. Most of the time is spent in the reduction phase anyway
because that scales exponentially in loop size, so this is not a bad option.

Also, should put a cap on how far into the network we go to compute the environment.
This is just a polynomial scaling problem but becomes significant with large networks.

'''


def norm(t):
    '''
    The L2 norm of the tensor. Must be a NetworkTensor.
    '''
    t2 = t.copy()[0]
    tens = t.contract(range(t.rank), t2, range(t.rank), elimLoops=False)
    tens.network.cutLinks()
    tens.contractRank2()

    return 0.5 * tens.logNorm

def envNorm(t, env):

    c = t.contract(range(t.rank), env, range(t.rank), elimLoops=False)
    c.newIDs()
    c = c.contract(range(t.rank), t, range(t.rank), elimLoops=False)

    c.network.cutLinks()
    c.contractRank2()

    return 0.5 * c.logNorm

class optimizer:
    def __init__(self, tensors, tolerance, environment, bids, otherbids):

        # Reorder externalBuckets to match the underlying ordering of nodes in the loop.
        # At the end of the day what we'll do is read out the tensors in the loop by
        # bucket index and slot them back into the full network in place of the
        # original tensors with the same bucket index. As a result we can feel free
        # to move around the buckets in externalBuckets so long as the buckets themselves
        # are unchanged. The key is that we never need to use the loop network
        # (which is a NetworkTensor) as a tensor proper.


        self.inds = list([0 for _ in range(tensors.rank)])
        buckets = [tensors.externalBuckets[0]]
        n = buckets[0].node
        prevNodes = set()
        prevNodes.add(n)
        while len(buckets) < len(tensors.externalBuckets):
            n = tensors.network.internalConnected(n).difference(prevNodes).pop()
            exb = list(b for b in n.buckets if b in tensors.externalBuckets)[0]
            buckets.append(exb)
            self.inds[tensors.externalBuckets.index(exb)] = len(prevNodes)
            prevNodes.add(n)
        tensors.externalBuckets = buckets

        # Contract the environment against itself
        inds = []
        for i in range(environment.rank):
            if environment.externalBuckets[i].id not in otherbids:
                inds.append(i)

        newEnv = environment.copy()[0]

        environment = environment.contract(inds, newEnv, inds, elimLoops=False)
        environment.contractRank2()

        # Replace environment with identity
        for n in environment.network.nodes:
            if n.tensor.rank == 2:
                n.tensor = ArrayTensor(np.identity(n.tensor.shape[0]))

        # Normalise environment
        envn = norm(environment)
        environment.externalBuckets[0].node.tensor = environment.externalBuckets[0].node.tensor.divideLog(envn)
        
        # There are two external buckets for each external on loop (one in, one out),
        # and these are now ordered in two sets which correspond, as in
        # [p,q,r,..., p,q,r,...]
        # Now we reorder these to match the order of nodes in the loop.
        buckets1 = []
        buckets2 = []
        extbids = list(b.id for b in environment.externalBuckets)
        for b in tensors.externalBuckets:
            ind = bids.index(b.id)

            other = otherbids[ind]
            ind = extbids.index(other)
            buckets1.append(environment.externalBuckets[ind])
            buckets2.append(environment.externalBuckets[ind + tensors.rank])

        environment.externalBuckets = buckets1 + buckets2

        # Normalize tensors
        self.norm = envNorm(tensors, environment)
        tensors.externalBuckets[0].node.tensor = tensors.externalBuckets[0].node.tensor.divideLog(self.norm)

        # Store inputs
        self.tensors = tensors
        self.environment = environment
        self.tolerance = tolerance

        # Construct guess
        x = optTensor(self.tensors, self.environment)

        err = self.tolerance
        while True:
            i = 0
            did = False
            while i < tensors.rank:
                y = deepcopy(x)
                reduced = y.reduce(i)
                if reduced:
                    y.optimizeSweep(err)
                    if (y.error < err):
                        print('Succeeded reduction. Error:', y.error)
                        x = y
                        did = True
                        err -= y.error
                i += 1
            if not did:
                break

        # Undo normalization
        temp = x.guess.externalBuckets[0].node.tensor
        temp = temp.multiplyLog(self.norm)
        x.guess.externalBuckets[0].node.tensor = temp
        
        self.x = x

def optimize(tensors, tolerance, environment, bids, otherbids):
    opt = optimizer(tensors, tolerance, environment, bids, otherbids)
    return opt.x.guess, opt.inds



