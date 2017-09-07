### Logging levels

levels = {}
levels['svd'] = 'debug'
levels['linalg'] = 'debug'
levels['misc'] = 'debug'
levels['arrays'] = 'debug'
levels['treeTensor'] = 'debug'
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

### Specifies whether to enable plotting

plot = False

### Determines the cutoff size below which matrices default to the dense SVD.

svdCutoff = 1e3

### Determines the maximum number of attempts for the interpolative SVD.

svdTries = 4
