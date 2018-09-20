import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.linalg import eigh

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['linalg'])

##################################
# General Linear Algebra Functions
##################################


def kroneckerDelta(dim, length):
    '''
    Returns the Kronecker delta of dimension dim and length length.
    Note that if dim == 0 the scalar 1. is returned. If dim == 1 a
    vector of ones is returned. These are the closest conceptual
    analogues of the Kronecker delta, and work in a variety of circumstances
    where a Kronecker delta is the natural higher-dimensional generalisation.
    '''
    if dim == 0:
        return 1
    elif dim == 1:
        return np.ones(length)
    elif dim > 1:
        arr = np.zeros(tuple(length for i in range(dim)))
        np.fill_diagonal(arr, 1.0)
        return arr


def adjoint(m):
    return np.transpose(np.conjugate(m))

def linear_solve(a, b):
    try:
        res = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        logger.warning('Linear solve failed. Falling back on least squares.')
        if len(b.shape) == 2:
            logger.warning('Using least squares on matrix requires looping over columns, which is slow.')
            vecs = list(b[:,i] for i in range(b.shape[1]))
        else:
            vecs = [b]

        ress = []
        for v in vecs:        
            ret = lsqr(a, v, atol=1e-14, btol=1e-14, iter_lim=1e4, conlim=1e20)
            res = ret[0]
            istop = ret[1]
            if istop == 1 or istop == 4:
                logger.debug('Least squares found an exact solution.')
            elif istop == 2 or istop == 5:
                logger.debug('Least squares solve proceeded to desired tolerance.')
            elif istop == 3 or istop == 6:
                logger.warning('Least squares exited prematurely due to ill-conditioning.')
                logger.warning('LSQR L2 Norm Error: ' + str(ret[4]))
            elif istop == 7:
                logger.warning('Least squares exited due to iteration limit.')
                logger.warning('LSQR L2 Norm Error: ' + str(ret[4]))
            ress.append(res)
        if len(b.shape) == 1:
            res = ress[0]
        else:
            res = np.array(ress).T
    return res

def sqrtm_psd(a):
    w, v = eigh(a)
    w[w < np.max(w) * config.runParams['epsilon']] = 0 # Filter out negative floating point noise
    w = np.sqrt(w)
    return (v * w).dot(v.conj().T)
