# Logging levels

levels = {}
levels['svd'] = 'debug'
levels['linalg'] = 'debug'
levels['misc'] = 'debug'
levels['arrays'] = 'debug'
levels['treeTensor'] = 'debug'
levels['treeNetwork'] = 'debug'
levels['identityTensor'] = 'debug'
levels['bucket'] = 'debug'
levels['link'] = 'debug'
levels['compress'] = 'debug'
levels['mergeLinks'] = 'debug'
levels['latticeNode'] = 'debug'
levels['network'] = 'debug'
levels['networkTree'] = 'debug'
levels['node'] = 'debug'
levels['priorityQueue'] = 'debug'
levels['tensor'] = 'debug'
levels['compress'] = 'debug'
levels['arrayTensor'] = 'debug'
levels['traceMin'] = 'debug'

levels['mergeContractor'] = 'info'
levels['generic'] = 'info'

# Determines the cutoff size below which matrices default to the dense SVD.

svdCutoff = 1e3

# Determines the maximum number of attempts for the interpolative SVD.

svdTries = 4

# Determines the maximum bond dimension for using sparse SVD. Written as a
# fraction of the matrix rank.

svdBondCutoff = 0.1

# Sets an upper bound on memory usage

mem_limit = 2**33
