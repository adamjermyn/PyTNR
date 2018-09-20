import numpy as np
from numpy.linalg import svd
from scipy.linalg.interpolative import svd as svdI
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds
from itertools import combinations

from TNR.Utilities.arrays import permuteIndices
from TNR.Utilities.linalg import adjoint, linear_solve, sqrtm_psd

from TNR.Utilities.logger import makeLogger
from TNR import config
logger = makeLogger(__name__, config.levels['svd'])

###################################
# Linear Operator and SVD Functions
###################################


def matrixProductLinearOperator(matrix1, matrix2):
    '''
    The reason we implement our own function here is that the dot product
    associated with the standard LinearOperator class has an extremely slow
    type-checking stage which has to be performed every time a product is calculated.
    '''

    if matrix1.shape[0] < matrix1.shape[1] or matrix2.shape[1] < matrix2.shape[0]:
        return np.dot(matrix1, matrix2)

    shape = (matrix1.shape[0], matrix2.shape[1])

    def matvec(v):
        return np.dot(matrix1, np.dot(matrix2, v))

    def matmat(m):
        return np.dot(matrix1, np.dot(matrix2, m))

    def rmatvec(v):
        return np.dot(np.transpose(matrix2), np.dot(np.transpose(matrix1), v))

    return LinearOperator(shape, matvec=matvec, matmat=matmat, rmatvec=rmatvec)


def compareSVD(matrix, u, s, v):
    '''
    This method compares a matrix against its singular value decomposition
    and returns the relative L2 error.

    The arguments are:
            matrix		-	The matrix.
            u		-	The unitary matrix U in the SVD.
            s		-	The diagonal matrix S in the SVD.
            v		-	The unitary matrix V^adjoint in the SVD.
    '''
    return np.sum(np.abs(np.einsum('ij,j,jk->ik', u, s, v) -
                         matrix)**2) / np.sum(np.abs(matrix)**2)


def environmentSVD(matrix, environmentLeft, environmentRight, precision):
    '''
    Computes the SVD of matrix with respect to multiplication on the left by
    environmentLeft and on the right by environmentRight.
    
    :param matrix: 
    :param environmentLeft: 
    :param environmentRight: 
    :param precision: 
    :return: 
    '''
    
    environmentLeft = sqrtm_psd(environmentLeft)
    environmentRight = sqrtm_psd(environmentRight)

    mat = np.dot(environmentLeft, matrix)
    mat = np.dot(mat, environmentRight)
    
    u, s, v = sortSVD(np.linalg.svd(mat))
    
    s = s[::-1]
    cp = np.cumsum(s**2) / np.sum(s**2)
    s = s[::-1]

    ind = np.searchsorted(cp, precision, side='left')
    ind = len(cp) - ind

    u = u[:, :ind]
    s = s[:ind]
    v = v[:ind, :]
    
    us = np.einsum('ij,j->ij',u,np.sqrt(s))
    vs = np.einsum('ij,j->ij',np.conjugate(np.transpose(v)),np.sqrt(s))

    A = linear_solve(environmentLeft, us)
    B = linear_solve(environmentRight, vs)
        
    return A, B

def sortSVD(decomp):
    '''
    This method sorts the singular values of an SVD.
    If provided the U and V matrices are permuted accordingly.
    This method takes as input:
            decomp - Either an array of singular values or a tuple of the form (U, S, V), where S is the array of singular values.

    The returned data is in the same format as the input.
    '''

    if isinstance(decomp, np.ndarray):
        inds = np.argsort(decomp)
        inds = inds[::-1]
        ret = decomp[inds]
    else:
        if len(decomp) != 3:
            raise ValueError(
                'Unknown input format. Not a numpy array and not of the form (U,S,V).')
        u, s, v = decomp
        inds = np.argsort(s)
        inds = inds[::-1]
        u = u[:, inds]
        s = s[inds]
        v = v[inds, :]
        ret = (u, s, v)

    return ret


def svdByPrecision(matrix, precision, compute_uv):
    '''
    This method wraps various SVD methods to provide a unified interface for computing the
    SVD to a specified precision. It returns the SVD with the singular values sorted in
    descending order, which some solvers do not guarantee.

    The arguments are:
            matrix		-	A 2D array
            precision	-	This is a float in the range [0,1) specifying
                                    the relative precision of the desired decomposition.
            compute_uv	-	A bool specifying whether or not to compute the matrices
                                    U and V or just the singular values. Note that some methods
                                    intrinsically compute all of these, so when those are used
                                    this just determines whether or not U and V are returned.

    For small matrices the default is to use the dense SVD implementation found in NumPy.
    For larger matrices iterative methods are tried first and compared against the desired precision.
    These may be applied repeatedly with different arguments until either the decomposition converges,
    an error is found, or it becomes more performant to move on to another method.
    Large matrices then fall back on the dense SVD.

    The cutoff between large and small here is specified in the config file.
    '''
    if precision < 0:
        raise ValueError(
            'Precision cannot be negative. Specified: ' +
            str(precision) +
            '.')
    if precision >= 1:
        raise ValueError(
            'Precision cannot be greater than 1. Specified: ' +
            str(precision) +
            '.')
    if np.sum(1 - np.isfinite(matrix)) > 0:
        raise ValueError(
            'Cannot decompose a matrix with infinite or NaN elements.')

    # The dense decomposition is more efficient for small matrices.
    if matrix.size < config.svdCutoff:
        decomp = svd(matrix, full_matrices=False, compute_uv=compute_uv)
    else:
        # First try the interpolative decomposition SVD. This typically
        # performs very well.
        try:
            u, s, v = svdI(matrix, precision)
            v = np.conjugate(np.transpose(v))
            tries = 0
            error = compareSVD(matrix, u, s, v)

            # If the error is not below the requested precision, try again with
            # an artificially more precise request.
            while error > precision and tries < config.svdTries:
                logger.debug(
                    'Interpolative SVD did not reach required precision. Actual: ' +
                    str(error) +
                    '. Requested: ' +
                    str(precision) +
                    '. Retrying with more precise request.')
                tries += 1
                u, s, v = svdI(matrix, precision / 2**tries)
                v = np.conjugate(np.transpose(v))
                error = compareSVD(matrix, u, s, v)

            if error > precision:
                # Getting to this stage means the interpolative decomposition just isn't working.
                # We now fall back on the dense decomposition.
                decomp = svd(
                    matrix,
                    full_matrices=False,
                    compute_uv=compute_uv)
            else:
                if compute_uv:
                    decomp = (u, s, v)
                else:
                    decomp = s

        except BaseException:
            # Means the SVD has raised an error so we fall back on the dense
            # one.
            decomp = svd(matrix, full_matrices=False, compute_uv=compute_uv)

    decomp = sortSVD(decomp)
    return decomp


def svdByRank(matrix, rank, compute_uv):
    '''
    This method wraps various SVD methods to provide a unified interface. It returns the SVD
    with the singular values sorted in descending order, which some solvers do not guarantee.

    The arguments are:
            matrix		-	A 2D array
            precision	-	This is an integer >= 1 giving the bond dimension of the decomposition.
            compute_uv	-	A bool specifying whether or not to compute the matrices
                                    U and V or just the singular values. Note that some methods
                                    intrinsically compute all of these, so when those are used
                                    this just determines whether or not U and V are returned.

    For small matrices the default is to use the dense SVD implementation found in NumPy.
    For larger matrices iterative methods are tried first and compared against the desired precision.
    The cutoff between large and small here is specified in the config file.
    '''
    if rank < 1:
        raise ValueError(
            'Rank cannot be zero or negative. Specified: ' +
            str(rank) +
            '.')
    if np.sum(1 - np.isfinite(matrix)) > 0:
        raise ValueError(
            'Cannot decompose a matrix with infinite or NaN elements.')

    # In this case the precision is an integer giving the desired bond
    # dimension.
    if matrix.size < config.svdCutoff or rank > config.svdBondCutoff * \
            min(matrix.shape):
        # The dense decomposition is more efficient for small matrices and for
        # high bond dimension.
        decomp = svd(matrix, full_matrices=False, compute_uv=compute_uv)
    else:  # The regular sparse decomposition is more efficient for large matrices at low bond dimension.
        decomp = svds(
            matrix,
            k=rank,
            which='LM',
            return_singular_vectors=compute_uv)

    decomp = sortSVD(decomp)
    return decomp


def entropy(array, pref=None, tol=1e-3):
    '''
    This method determines the best pair of indices to split off.
    That pair is just the one with the minimum entropy to the rest of the indices.

    This is determined by iteratively refining bounds of the entropy associated with
    each possible pair and throwing away provably worse options until only one remains.

    To avoid refining endlessly when options are essentially identical this method
    takes as optional input a tolerance tol below which it does not care about
    entropy differences. This defaults to 1e-3 and is treated as an absolute tolerance.
    That is, the returned answer is guaranteed to be optimal within this tolerance.

    This method also takes as an optional input pref, which specifies a tie-breaking
    preference in cases where multiple options lie within tol of each other and the
    optimum. This should be specified as a set containing a pair of indices.
    '''

    # Make sure pref is a set:
    if pref is None:
        pref = set()
    else:
        pref = set(pref)

    # Generate list of pairs of indices
    indexLists = list(combinations(range(len(array.shape)), 2))

    # We filter out options which are complements of one another, and
    # hence give the same answer. We do not filter out pref in this process.

#	print('Filtering complements.')
    indexLists = [set(q) for q in indexLists]

    complements = [set(range(len(array.shape))).difference(l)
                   for l in indexLists]
    indexSets = [set(l) for l in indexLists]
    while len(complements) > 0:
        c = complements.pop()
        if c in indexSets and c != pref:
            indexSets.remove(c)
            s = set(range(len(array.shape)))
            s = s.difference(c)
            if s in complements:
                complements.remove(s)
    indexLists = [tuple(l) for l in indexSets]

#	print('Examining options.')

    # Lists for storing intermediate results.
    mins = [1e10 for _ in indexLists]			# Lower bound on entropy
    maxs = [-1 for _ in indexLists]				# Upper bound on entropy
    # Frobenius norms of the array in different shapes
    norms = [-1 for _ in indexLists]
    # Temporary storage for singular values
    knownVals = [None for _ in indexLists]
    # Stores the indices of options which have not been ruled out.
    liveIndices = list(range(len(indexLists)))

    # We start with just two singular values (set to 1 so that it becomes 2
    # upon doubling)
    bondDimension = 1
    # Keeps track of the index with the lowest upper bound on the entropy
    lowest = [1e10, -1]
    while len(liveIndices) > 1:
        bondDimension *= 2  # Double the bond dimension

        for i in list(
                liveIndices):  # We copy the list so we can remove from it while looping
            indices = indexLists[i]

            # Put the array in the right shape
            arr = permuteIndices(array, indices)
            sh = arr.shape[:len(indices)]
            s = np.product(sh)
            arr = np.reshape(arr, (s, -1))

            # Calculate the norm if it hasn't been done already
            if norms[i] == -1:
                norms[i] = np.linalg.norm(arr)

            # Take advantage of rank bounds
            mat = np.copy(arr)
            if arr.shape[0] > arr.shape[1]:
                mat = np.transpose(mat)
            mat = np.dot(mat, np.transpose(mat))

            # If the bond dimension is too large, full SVD is required.
            lams = svdByRank(mat, bondDimension, False)
            lams = np.sqrt(lams)
            lams /= norms[i]					# Normalize
            knownVals[i] = lams**2				# Turn into probabilities
            p = knownVals[i]
            # Ensure probabilities are non-zero for floating point reasons
            p = p[p > 0]
            # Compute entropy of probabilities
            mins[i] = -np.sum(p * np.log(p))
            maxs[i] = mins[i]

            # If there is left-over probability we get additional entropy,
            # but we don't know how much so we just calculate bounds.
            q = 1 - np.sum(p)
            if q > 0 and bondDimension < min(arr.shape):
                # Corresponds to a single singular value holding the remaining
                # probability
                mins[i] -= q * np.log(q)
                # Corresponds to multiple singular values holding it
                maxs[i] -= q * np.log(q / (min(arr.shape) - bondDimension))

            # Now we check if any can be eliminated
            if maxs[i] < lowest[0] - \
                    tol:		# Means this is better than the previous best
                lowest[0] = maxs[i]
                lowest[1] = i
            # Means that this is strictly worse than the current best
            elif mins[i] > lowest[0] + tol:
                liveIndices.remove(i)
            else:								# Means this is tied within tolerance to the current best
                if pref == set():				# If we have no preference we remove this unless it is the current best
                    if i != lowest[1]:
                        liveIndices.remove(i)
                elif pref != set(indexLists[i]) and pref in [indexSets[j] for j in liveIndices]:
                                                                                # Otherwise we remove this so long as the preferred option is still live
                                                                                # and
                                                                                # this
                                                                                # is
                                                                                # not
                                                                                # it.
                    liveIndices.remove(i)
                else:
                    # Means the preferred option is still live and is tied for
                    # best.
                    return list(indexLists[i])
#			print(mins, maxs, lowest, i, pref, indexLists)
    return list(indexLists[liveIndices[0]])


def splitArray(array, indices, accuracy=1e-4):
    perm = []

    sh1 = [array.shape[i] for i in indices]
    sh2 = [array.shape[i] for i in range(len(array.shape)) if i not in indices]
    indices1 = list(indices)
    indices2 = [i for i in range(len(array.shape)) if i not in indices]

    arr = permuteIndices(array, indices)
    arr = np.reshape(arr, (np.product(sh1), np.product(sh2)))
    u, lam, v = svdByPrecision(arr, accuracy, True)

    p = lam**2
    p /= np.sum(p)
    cp = np.cumsum(p)

    ind = np.searchsorted(cp, accuracy, side='left')
    ind = len(cp) - ind

    u = u[:, :ind]
    lam = lam[:ind]
    v = v[:ind, :]

    u *= np.sqrt(lam)[np.newaxis, :]
    v *= np.sqrt(lam)[:, np.newaxis]

    u = np.reshape(u, sh1 + [ind])
    v = np.reshape(v, [ind] + sh2)

    return u, v, indices1, indices2
