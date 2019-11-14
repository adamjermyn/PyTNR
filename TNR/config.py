import yaml
import os.path
from pathlib import Path

# Logging levels

levels = {}
levels['svd'] = 'debug'
levels['linalg'] = 'debug'
levels['misc'] = 'debug'
levels['arrays'] = 'debug'
levels['treeTensor'] = 'info'
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
levels['tensor'] = 'info'
levels['compress'] = 'debug'
levels['arrayTensor'] = 'info'
levels['traceMin'] = 'debug'

levels['mergeContractor'] = 'info'
levels['generic'] = 'info'

# Run parameters
runParams = {}

# Determines the cutoff size below which matrices default to the dense SVD.

runParams['svdCutoff'] = 3e2

# Determines the maximum number of attempts for the interpolative SVD.

runParams['svdTries'] = 4

# Determines the maximum bond dimension for using sparse SVD. Written as a
# fraction of the matrix rank.

runParams['svdBondCutoff'] = 0.1

# Sets an upper bound on memory usage

runParams['mem_limit'] = 2**33

# Read config file if possible

home = str(Path.home())
config_path = home + '/.tnr_config'
config_file = Path(config_path)
if config_file.is_file():
	data = yaml.load(open(config_path, 'r'))
	if 'levels' in data.keys():
		for key in data['levels'].keys():
			levels[key] = data['levels'][key]
	if 'runParams' in data.keys():
		for key in data['runParams'].keys():
			runParams[key] = data['runParams'][key]

svdMaxSize = 5e8
svdCutoff = int(runParams['svdCutoff'])
svdTries = int(runParams['svdTries'])
svdBondCutoff = float(runParams['svdBondCutoff'])
mem_limit = int(runParams['mem_limit'])
