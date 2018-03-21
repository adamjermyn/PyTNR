import numpy as np

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
