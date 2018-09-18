import numpy as np
from copy import deepcopy
from scipy.sparse.linalg import lsqr

from TNR.Tensor.arrayTensor import ArrayTensor

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['treeTensor'])

def norm(t):
    '''
    The L2 norm of the tensor. Must be a NetworkTensor.
    '''
    t2 = t.copy()
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

def rank1guess(base, env):
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
    n.tensor = n.tensor.divideLog(envNorm(start, env))

    return start


class optTensor:
    def __init__(self, loop, environment):
        # Loop and guess must have the same bucket ID's.
        # Environment specifies a tensor against whose contraction guess should be optimised.
        # That is, if E is the environment, T is the loop and G is the guess, we're minimising
        # |T.E - G.E|^2 = (T.E)^2 + (G.E)^2 - 2(T.E).(G.E)
        # Taking the gradient with respect to G yields
        # Grad = E (G.E) - 2 (T.E)._.E
        self.environment = environment
        self.loop = loop
        self.loopNorm = np.exp(2*envNorm(self.loop, self.environment)) # This is invariant.
        self.guess = rank1guess(loop, self.environment)
        self.ranks = tuple([1 for _ in range(len(self.loop.externalBuckets))])
        self.rands = list([self.random() for _ in range(20)])

    @property
    def guessNorm(self):
        return np.exp(2*envNorm(self.guess, self.environment))

    @property
    def error(self):
        t1 = self.loop.copy()
        t2 = self.guess.copy()
        
        c1 = t1.contract(range(t1.rank), self.environment, range(t1.rank), elimLoops=False)

        c = c1.contract(range(c1.rank), t2, range(c1.rank), elimLoops=False)

        return self.loopNorm + self.guessNorm - 2 * c.array

    def __str__(self):
        return str(self.ranks)

    def __len__(self):
        return self.loop.rank

    def random(self):
        t = self.loop.copy()
        for n in t.network.nodes:
            n.tensor = ArrayTensor(np.random.randn(*n.tensor.shape))
        return t

    def prepareEnv(self, index):
        '''
        We define two tensors, N and W.

        N is given by contracting t2 against itself except for the tensors at the specified
        index. The full environment tensor is used. 
        This yields a rank-6 object, where two indices arise from the environment.

        W is then given by contracting all of t2 (other than t2[index]) against t1 with
        the full environment in between. This yields a rank-3 object.

        We then let

        N . t2[index] = W

        and solve for t2[index]. This is readily phrased as a matrix problem by
        flattening N along all indices other than that associated with t2[index],
        and doing the same for W.
        '''

        # Make W
        t1 = self.loop.copy()
        t2 = deepcopy(self.guess)
        n = t2.externalBuckets[index].node
        t = t1.contract(range(t1.rank), self.environment, range(t1.rank), elimLoops=False)
        t = t.contract(range(t1.rank), t2, range(t1.rank), elimLoops=False)
        n = list(m for m in t.network.nodes if m.id == n.id)[0]
        t.removeNode(n)
        W = t

        # Make N
        t1 = t2  # We can reuse t2 because we haven't changed it.
        t2 = self.guess.copy()

        # Get nodes
        n1 = t1.externalBuckets[index].node
        n2 = t2.externalBuckets[index].node

        # Contract environment onto t2
        t2 = t2.contract(range(t2.rank), self.environment, range(t2.rank), elimLoops=False)

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
        N, W = self.prepareEnv(index)

        N_bak = np.array(N)
        W_bak = np.array(W)
        # Flatten, solve, unflatten

        sh = W.shape
        W = np.reshape(W, (-1,))
        N = np.reshape(N, (len(W), len(W)))
        res = lsqr(N, W)[0]
        res = np.reshape(res, sh)

        local_norm = np.einsum('ijk,ijklmn,lmn->',res,N_bak,res)
        err = local_norm + self.loopNorm - 2*np.einsum('ijk,ijk->',res,W_bak)
        try:
            self.guess.externalBuckets[index].node.tensor = ArrayTensor(res)
        except:
            print(norm(self.environment))
            print(N)
            print(W)
            print(res)
            exit()

        return err, local_norm
    def optimizeSweep(self, stopErr, stop=0.01):
        # Optimization loop
        dlnerr = 1
        err1 = 1e100

        while dlnerr > stop and err1 > stopErr:
            for i in range(self.loop.rank):
                self.optimizeIndex(i)
            err2 = self.error
            derr = (err1 - err2)
            dlnerr = derr / err1
            err1 = err2
            logger.debug('Error: ' + str(err1) + ', ' + str(dlnerr))

        return err1

    def reduce(self, index):
        '''
        Expands the dimension of the bond between the tensors in t attached to
        index and index+1 by one. The final entry along this dimension is simply deleted.
        If this results in a singular network the entries are changed randomly until
        this is not the case.
        '''

        # Get the nodes
        n1 = self.guess.externalBuckets[index].node
        n2 = self.guess.externalBuckets[(index + 1) % self.guess.rank].node

        # Get the tensors
        t1 = n1.tensor
        t2 = n2.tensor

        # Identify the indices for expansion
        assert n2 in n1.connectedNodes
        assert n1 in n2.connectedNodes
        i1 = n1.indexConnecting(n2)
        i2 = n2.indexConnecting(n1)

        sh = list(t1.shape)
        shq = list(t2.shape)

        # Reduce the first tensor
        sh2 = list(sh)
        sh2[i1] -= 1
        arr = t1.array
        sl = [slice(0, sh2[j]) for j in range(len(sh2))]
        arr = arr[sl]
        n1.tensor = ArrayTensor(arr)

        # Reduce the second
        sh2 = list(shq)
        sh2[i2] -= 1
        arr = t2.array
        sl = [slice(0, sh2[j]) for j in range(len(sh2))]
        arr = arr[sl]
        n2.tensor = ArrayTensor(arr)

        nor = np.sqrt(norm(self.guess))
        
        while abs(nor) < 1e-5: # Keep trying until the tensor isn't singular
            arr = n1.tensor.array
            arr += np.random.randn(*arr.shape)
            n1.tensor = ArrayTensor(arr)
            arr = n2.tensor.array
            arr += np.random.randn(*arr.shape)
            n2.tensor = ArrayTensor(arr)

            nor = np.sqrt(norm(self.guess))

        temp = self.guess.externalBuckets[0].node.tensor.array
        if np.isfinite(nor) and nor > 1e-5:
            temp /= nor
        self.guess.externalBuckets[0].node.tensor = ArrayTensor(temp)

        self.ranks = list(self.ranks)
        self.ranks[index] -= 1
        self.ranks = tuple(self.ranks)


    def expand(self, index, fill='random', amount=1):
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
            sh2[i1] += amount
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
            sh2[i2] += amount
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
            if np.isfinite(nor) and nor > 1e-5:
                temp /= nor
            self.guess.externalBuckets[0].node.tensor = ArrayTensor(temp)

        self.ranks = list(self.ranks)
        self.ranks[index] += amount
        self.ranks = tuple(self.ranks)




