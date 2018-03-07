import numpy as np
from copy import deepcopy
from scipy.sparse.linalg import LinearOperator, bicgstab, lsqr

from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.Network.bucket import Bucket

def shift(l, n):
    '''
    Shifts the list l forward by n indices.
    '''

    n = n % len(l)
    return l[-n:] + l[:-n]

def contract(t1, ind1, t2, ind2):
    '''
    Contracts two NetworkTensors against one another along external buckets.
    '''
    return t1.contract(ind1, t2, ind2, elimLoops=False)

def norm(t1):
    '''
    Returns the L2 norm of the specified NetworkTensor.
    '''

    t2 = t1.copy()
    t3 = t1.copy()

    return contract(t2, list(range(t1.rank)), t3, list(range(t1.rank))).array

def cost(t):
    '''
    A cost function for weighting options when optimizing loops.
    '''
    return t.compressedSize

def expand(t, index, fill='random'):
    '''
    Assumes that the external indices are ordered such that neighbouring (in the periodic sense)
    external indices are attached to neighbouring tensors in the network.

    Expands the dimension of the bond between the tensors in t attached to
    index and index+1 by one. The new matrix elements are filled as specified by fill:
        'random' - Numbers drawn at random from a unit-variance zero-mean normal distribution.
        'zero' - Zeros.
    '''
    # Copy the tensor
    t = t.copy()

    # Get the nodes
    n1 = t.externalBuckets[index].node
    n2 = t.externalBuckets[(index + 1) % t.rank].node

    # Get the tensors
    t1 = t.externalBuckets[index].node.tensor
    t2 = t.externalBuckets[(index + 1) % t.rank].node.tensor

    # Identify the indices for expansion
    assert n2 in n1.connectedNodes
    assert n1 in n2.connectedNodes
    i1 = n1.indexConnecting(n2)
    i2 = n2.indexConnecting(n1)

    # Expand the first tensor
    sh = list(t1.shape)
    sh2 = list(sh)
    sh2[i1] += 1
    if fill == 'random':
        arr = np.random.randn(*sh)
    elif fill == 'zeros':
        arr = np.zeros(sh)
    else:
        raise ValueError('Invalid fill prescription specified.')
    sl = [slice(0,sh[j]) for j in range(len(sh))]
    arr[sl] = t1.array
    t.externalBuckets[index].node.tensor = ArrayTensor(arr)

    # Expand the second tensor
    sh = list(t2.shape)
    sh2 = list(sh)
    sh2[i2] += 1
    if fill == 'random':
        arr = np.random.randn(*sh)
    elif fill == 'zeros':
        arr = np.zeros(sh)
    else:
        raise ValueError('Invalid fill prescription specified.')
    sl = [slice(0,sh[j]) for j in range(len(sh))]
    arr[sl] = t2.array
    t.externalBuckets[(index + 1) % t.rank].node.tensor = ArrayTensor(arr)

    return t


def prepareNW(t1, t2, index):
    '''

    Following equation S10 in arXiv:1512.04938, we construct the operators N and W.

    N is given by removing the node at index from t2 and contracting it against itself.
    W is given by removing the node at index from t2 and contracting it against t1.
    ''' 

    # Copy t2 and remove the node at index
    t2new = t2.copy()
    sh = t2new.externalBuckets[index].node.tensor.shape
    t2new.removeNode(t2new.externalBuckets[index].node)

    # Contract against t1 to generate W.
    red = list(range(t1.rank))
    red.remove(index)

    w = contract(t1, red, t2new, range(t2.rank - 1))

    # Contract t2 against itself to generate N.
    n = contract(t2new, range(t2.rank - 1), t2new.copy(), range(t2.rank - 1))

    # Bolt on the identity
    iden = np.identity(t2.externalBuckets[index].size)
    n.addTensor(ArrayTensor(iden))

    # Construct array versions of W and N
    arrW = w.array
    arrN = n.array

    # The indices are automatically aligned correctly by construction.

    return arrN, arrW, n, w, sh

def optimizeTensor(t1, t2, index):
    '''
    t1 and t2 are tensors representing loops which are to be contracted against one another.
    Their external buckets must have corresponding ID's (e.g. they must be formed from
    copying the same original tensor).

    The return value is a version of t2 which maximizes the inner product of t1 and t2, subject
    to the constraint norm(t1) == norm(t2) == 1. In the optimization process only
    the tensor attached to the specified external index may be modified.

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

    # Now we construct N and W.

    for b in t1.externalBuckets:
        assert not b.linked
    for b in t2.externalBuckets:
        assert not b.linked

    N, W, n, w, sh = prepareNW(t1, t2, index)

    # Reshape into matrices
    W = np.reshape(W, (-1,))
    N = np.reshape(N, (len(W), len(W)))

    try:
        res = np.linalg.solve(N, W)
    except np.linalg.linalg.LinAlgError:
        res = lsqr(N, W)[0]

    ret = deepcopy(t2)

    # Un-flatten and put res into ret at the appropriate place.

    # For some reason the norm of the return loop is always exactly 0.5...

    res = np.reshape(res, sh)
    ret.externalBuckets[index].node.tensor = ArrayTensor(res)

    err = 2 * (1 - contract(t1, range(t1.rank), ret, range(t1.rank)).array)

    print(err, np.sum(t1.array**2),norm(t1), np.sum(ret.array**2), norm(ret))


    return ret, err

def optimizeRank(tensors, ranks, start, stop=0.1):
    '''
    tensors is a list of rank-3 tensors set such that the last index of each contracts
    with the first index of the next, and the last index of the last tensor contracts
    with the first index of the first one.

    The return value is an optimized list of tensors representing the original as best
    as possible subject to the constraint that the bond dimensions of the returned tensors
    are given by the entries in ranks, the first of which gives the dimension between the
    first and second tensors. The current approximation error is also returned.

    The ending criterion is given by the parameter stop, which specifies the minimum logarithmic
    change in the error between optimization sweeps (once it drops below this point the algorithm halts).

    A starting point for optimization may be provided using the option start.
    '''

    # Generate random starting point and normalize.
    t2 = deepcopy(start)

    n = next(iter(t2.network.nodes))
    temp = n.tensor.array
    temp /= np.sqrt(norm(t2))
    n.tensor = ArrayTensor(temp)

    # Optimization loop
    dlnerr = 1
    err1 = 1e100

    while dlnerr > stop:
        for i in range(tensors.rank):
            t2, err2 = optimizeTensor(tensors, t2, i)
        derr = (err1 - err2)
        dlnerr = derr / err1
        err1 = err2

    return t2, err1

#def kronecker(dim):
#   x = np.zeros((dim, dim, dim))
#   for i in range(dim):
#       x[i,i,i] = 1
#   return x

#def test(dim):
#   x = 5 + np.random.randn(dim,dim,dim)
#   return x

#def testTensors(dim, num):
    # Put no correlations in while maintaining a non-trivial physical index
#   tens = np.random.randn(dim, dim, dim)
#   for i in range(dim-1):
#       tens[:,i,:] = np.random.randn()

    # Apply random unitary operators.
#   from scipy.stats import ortho_group
#   tensors = [np.copy(tens) for _ in range(num)]
#   for i in range(num):
#       u = ortho_group.rvs(dim)
#       uinv = np.linalg.inv(u)
#       tensors[i] = np.einsum('ijk,kl->ijl', tensors[i], u)
#       tensors[(i+1)%num] = np.einsum('li,ijk->ljk', uinv, tensors[i])

#   return tensors

#def test2(dimRoot):
    # Constructs a tensor list which, upon contraction, returns 0 when
    # all indices are zero and 1 otherwise.

#   dim = dimRoot**2

#   t = np.ones((dim,dim,dim))
#   t[0,0,0] = 0

    # SVD
#   t = np.reshape(t, (dim**2, dim))
#   u, s, v = np.linalg.svd(t, full_matrices=False)
#   q = np.dot(u, np.diag(np.sqrt(s)))
#   r = np.dot(np.diag(np.sqrt(s)), v)

    # Insert random unitary between q and r
#   from scipy.stats import ortho_group
#   u = ortho_group.rvs(dim)
#   uinv = np.linalg.inv(u)
#   q = np.dot(q, u)
#   r = np.dot(uinv, r)

    # Decompose bond
#   q = np.reshape(q,(dim, dim, dimRoot, dimRoot))
#   r = np.reshape(r,(dimRoot, dimRoot, dim))

    # Split q
#   q = np.swapaxes(q, 1, 2)

#   q = np.reshape(q, (dim*dimRoot, dim*dimRoot))
#   u, s, v = np.linalg.svd(q, full_matrices=False)

#   a = np.dot(u, np.diag(np.sqrt(s)))
#   b = np.dot(np.diag(np.sqrt(s)), v)

#   a = np.reshape(a, (dim, dimRoot, dim*dimRoot))
#   b = np.reshape(b, (dim*dimRoot, dim, dimRoot))

#   print(np.einsum('ijk,kml,jlw->imw',a,b,r))


#   return [a,b,r]

#dimRoot = 5

#dim = dimRoot**2
#tensors = test2(dimRoot)

#a,b,r = tensors

#print(np.linalg.svd(np.einsum('ijk,kml->ijml',a,b).reshape((dim*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',b,r).reshape((dim**2*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',r,a).reshape((dimRoot**2,dim*dimRoot**2)))[1])

#tensors = [test(5) for _ in range(5)]
#tensors = testTensors(5, 5)
#tensors[0] /= np.sqrt(norm(tensors))
#print(optimize(tensors, 1e-5)[:2])
    

#print(np.linalg.svd(np.einsum('ijk,kml->ijml',a,b).reshape((dim*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',b,r).reshape((dim**2*dimRoot,dim*dimRoot)))[1])
#print(np.linalg.svd(np.einsum('ijk,kml->ijml',r,a).reshape((dimRoot**2,dim*dimRoot**2)))[1])

