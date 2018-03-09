import numpy as np
from copy import deepcopy
from scipy.sparse.linalg import lsqr

from TNR.Tensor.arrayTensor import ArrayTensor

def norm(t):
    '''
    Returns the L2 norm of the tensor.
    '''
    t1 = t.copy()
    t2 = t.copy()

    return t1.contract(range(t.rank), t2, range(t.rank), elimLoops=False).array

def remove(t, index):
    '''
    Returns a copy of the tensor with new bucket ID's and node ID's with
    the node associated with the specified external index removed.

    Also returns a dictionary mapping from external bucket ID's on the
    input tensor to those on the output tensor.
    '''
    d = {}

    tNew = t.copy()
    n = t.externalBuckets[index].node
    nNew = tNew.externalBuckets[index].node
    for b,bNew in zip(*(n.buckets, nNew.buckets)):
        d[b.id] = bNew.id

    tNew.removeNode(nNew)
    return tNew, d

def rank1guess(base):
    x = tuple(1 for _ in range(len(base.externalBuckets)))
    start = deepcopy(base)
    for n in start.network.nodes:
        sh = []
        for i,b in enumerate(n.buckets):
            if b in start.externalBuckets:
                sh.append(n.tensor.shape[i])
            else:
                sh.append(1)

        n.tensor = ArrayTensor(np.random.randn(*sh))

    n = next(iter(start.network.nodes))
    n.tensor = ArrayTensor(n.tensor.array / np.sqrt(norm(start)))

    return start

class optTensor:
    def __init__(self, loop):
        # Loop and guess must have the same bucket ID's.
        self.loop = loop
        self.guess = rank1guess(loop)
        self.ranks = tuple([1 for _ in range(len(self.loop.externalBuckets))])

    @property
    def loopNorm(self):
        return norm(self.loop)

    @property
    def guessNorm(self):
        return norm(self.guess)

    @property
    def error(self):
        t1 = self.loop.copy()
        t2 = self.guess.copy()
        c = t1.contract(range(t1.rank), t2, range(t1.rank), elimLoops=False).array
        return 2*(1 - c)

    def __hash__(self):
        return hash(self.ranks)

    def __str__(self):
        return str(self.ranks)

    def __len__(self):
        return self.loop.rank


    def densityMatrix(self, index):
        t1 = self.loop.copy()
        t2 = self.loop.copy()

        ax = list(range(t1.rank))
        ax.remove(index)

        return t1.contract(ax, t2, ax, elimLoops=False).array

    def prepareNW(self, index):
        '''
        Following equation S10 in arXiv:1512.04938, we first compute two tensors: N and W.

        N is given by contracting all of t2 against itself except for the tensors at the specified index.
        This yields a rank-6 object, where two indices arise from the implicit identity present in the bond
        between the two copies of t2[index].
        
        W is given by contracting all of t2 (other than t2[index]) against t1. This yields a rank-3 object.

        We then let
        
        N . t2[index] = W

        and solve for t2[index]. This is readily phrased as a matrix problem by flattening N along all indices
        other than that associated with t2[index], and doing the same for W.
        '''

        # Make W
        t1 = self.loop.copy()
        t2 = deepcopy(self.guess)
        n = t2.externalBuckets[index].node
        t = t1.contract(range(t1.rank), t2, range(t1.rank), elimLoops=False)
        n = list(m for m in t.network.nodes if m.id == n.id)[0]

        t.removeNode(n)
        W = t


        # Make N
        t1 = deepcopy(self.guess)
        t2 = self.guess.copy()

        # Get nodes
        n1 = t1.externalBuckets[index].node
        n2 = t2.externalBuckets[index].node

        # Contract identity onto t2
        iden = ArrayTensor(np.identity(self.loop.externalBuckets[index].size))
        t2 = t2.contract([index], iden, [0], elimLoops=False)

        # Restore index
        bs = t2.externalBuckets[::]
        bs = bs[:index] + [bs[-1]] + bs[index:-1]
        t2.externalBuckets = bs

        # Contract
        t = t1.contract(range(t1.rank), t2, range(t1.rank), elimLoops=False)

        # Remove nodes
        n1 = list(m for m in t.network.nodes if m.id == n1.id)[0]
        n2 = list(m for m in t.network.nodes if m.id == n2.id)[0]

        t.removeNode(n1)
        t.removeNode(n2)

        N = t

        # Because N is symmetric, the first three external Buckets correspond in order
        # to the last three. Because those in W are formed by removing the same node,
        # those are in the same order.

        return N.array, W.array

    def optimizeIndex(self, index):
        N, W = self.prepareNW(index)

        # Flatten, solve, unflatten

        sh = W.shape
        W = np.reshape(W, (-1,))
        N = np.reshape(N, (len(W), len(W)))
        try:
            res = np.linalg.solve(N, W)
        except np.linalg.linalg.LinAlgError:
            res = lsqr(N, W)[0]
        res = np.reshape(res, sh)

        self.guess.externalBuckets[index].node.tensor = ArrayTensor(res)


    def optimizeSweep(self, stop=0.1):
        # Optimization loop
        dlnerr = 1
        err1 = 1e100

        while dlnerr > stop:
            for i in range(self.loop.rank):
                self.optimizeIndex(i)
            err2 = self.error
            derr = (err1 - err2)
            dlnerr = derr / err1
            err1 = err2

        return err1

    def expand(self, index, fill='random'):
        '''
        Assumes that the external indices are ordered such that neighbouring (in the periodic sense)
        external indices are attached to neighbouring tensors in the network.

        Expands the dimension of the bond between the tensors in t attached to
        index and index+1 by one. The new matrix elements are filled as specified by fill:
            'random' - Numbers drawn at random from a unit-variance zero-mean normal distribution.
            'zero' - Zeros.
        '''

        # Get the nodes
        n1 = self.guess.externalBuckets[index].node
        n2 = self.guess.externalBuckets[(index + 1) % self.guess.rank].node

        # Get the tensors
        t1 = self.guess.externalBuckets[index].node.tensor
        t2 = self.guess.externalBuckets[(index + 1) % self.guess.rank].node.tensor

        # Identify the indices for expansion
        assert n2 in n1.connectedNodes
        assert n1 in n2.connectedNodes
        i1 = n1.indexConnecting(n2)
        i2 = n2.indexConnecting(n1)

        sh = list(t1.shape)
        shq = list(t2.shape)

        nor = 0
        while abs(nor) < 1e-5: # Keep trying until the tensor isn't singular
            # Expand the first tensor
            sh2 = list(sh)
            sh2[i1] += 1
            if fill == 'random':
                arr = np.random.randn(*sh2)
            elif fill == 'zeros':
                arr = np.zeros(sh2)
            else:
                raise ValueError('Invalid fill prescription specified.')
            sl = [slice(0,sh[j]) for j in range(len(sh))]
            arr[sl] = t1.array
            self.guess.externalBuckets[index].node.tensor = ArrayTensor(arr)

            # Expand the second tensor
            sh2 = list(shq)
            sh2[i2] += 1
            if fill == 'random':
                arr = np.random.randn(*sh2)
            elif fill == 'zeros':
                arr = np.zeros(sh2)
            else:
                raise ValueError('Invalid fill prescription specified.')
            sl = [slice(0,shq[j]) for j in range(len(shq))]
            arr[sl] = t2.array
            self.guess.externalBuckets[(index + 1) % self.guess.rank].node.tensor = ArrayTensor(arr)

            nor = np.sqrt(norm(self.guess))
            temp = self.guess.externalBuckets[0].node.tensor.array
            temp /= nor
            self.guess.externalBuckets[0].node.tensor = ArrayTensor(temp)

        self.ranks = list(self.ranks)
        self.ranks[index] += 1
        self.ranks = tuple(self.ranks)




