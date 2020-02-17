
# Requirements

PyTNR requires Python v3.3 or above and NumPy, NetworkX and PyYaml.

# Using PyTNR

Programs using PyTNR typically consist of two phases.
In the first phase they construct a tensor network.
To begin, make a network as follows:
```python
from TNR.Network.network import Network
network = Network()
```
This initializes an empty tensor network.

To insert tensors into the network, construct an array representing the tensor of interest, construct a Tensor object, and use that to construct a Node object:
```python
from TNR.Network.node import Node
from TNR.Tensor.arrayTensor import ArrayTensor
import numpy as np
arr = np.random.randn(3, 3, 3, 2)
tens = ArrayTensor(arr)
node = Node(tens)
```

In general you should construct all of your nodes and link them up to each other before adding them to your network.
This is because the Network class automatically classifies internal and external links at the time that nodes are added.
Linking nodes is done as follows:
```python
# Assumes n1 and n2 are Nodes
l = Link(n1.bucket[0], n2.bucket[0])
```
The `.buckets` specification indicates which index of a given node to link.
The buckets are just a way of keeping track of indices, and are numbered in the same way as the indices of the tensor.
Once the nodes are linked appropriately, add them to the network:
```python
network.addNode(n1)
network.addNode(n2)
```

Finally, the network can be contracted.
This is done using a contraction manager and a contraction heuristic.
For instance,
```
# Assumes network is a Network
from TNRG.Contractors.mergeContractor import mergeContractor
from TNRG.Contractors.heuristics import utilHeuristic
accuracy = 1e-3
n = mergeContractor(network, accuracy, utilHeuristic, optimize=True)
```
That's all there is to it. Now `n` contains a tensor tree or collection of tensor trees representing the contraction of `network`.

If you have any questions don't hesitate to ask, and also please take a look at the provided examples (in Examples /).

# Configuring PyTNR

PyTNR comes with a default configuration controlling various details of the singular value decomposition, logging and memory limits. The defaults are:
```
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

# Run parameters
runParams = {}

# Determines the cutoff size below which matrices default to the dense SVD.

runParams['svdCutoff'] = 1e3

# Determines the maximum number of attempts for the interpolative SVD.

runParams['svdTries'] = 4

# Determines the maximum bond dimension for using sparse SVD. Written as a
# fraction of the matrix rank.

runParams['svdBondCutoff'] = 0.1

# Sets an upper bound on memory usage

runParams['mem_limit'] = 2**33
```
In order to override these defaults create a file `.tnr_config` in your home directory.
Then specify the configuration using `yaml` syntax as in
```
levels:
 svd: debug

runParams:
 svdTries: 4
```
and so on.

# Referencing PyTNR

PyTNR is free to use, but if you use it for academic purposes please include a citation to the two methods papers:

Automatic Contraction of Unstructured Tensor Networks - Adam S. Jermyn arXiv: 1709.03080
Efficient Decomposition of High - Rank Tensors - Adam S. Jermyn arXiv: 1708.07471

BibTex entries for these are included below:

```


@ARTICLE{2017arXiv170903080J,
         author = {{Jermyn}, A.~S.},
         title = "{Automatic Contraction of Unstructured Tensor Networks}",
         journal = {ArXiv e - prints},
         archivePrefix = "arXiv",
         eprint = {1709.03080},
         primaryClass = "physics.comp-ph",
         keywords = {Physics - Computational Physics, Condensed Matter - Statistical Mechanics, Condensed Matter - Strongly Correlated Electrons},
         year = 2017,
         month = sep,
         adsurl = {http: // adsabs.harvard.edu / abs / 2017arXiv170903080J},
         adsnote = {Provided by the SAO / NASA Astrophysics Data System}
         }
'''
'''
@ARTICLE{2017arXiv170807471J,
         author = {{Jermyn}, A.~S.},
         title = "{Efficient Decomposition of High-Rank Tensors}",
         journal = {ArXiv e - prints},
         archivePrefix = "arXiv",
         eprint = {1708.07471},
         primaryClass = "physics.comp-ph",
         keywords = {Physics - Computational Physics, Computer Science - Numerical Analysis},
         year = 2017,
         month = aug,
         adsurl = {http: // adsabs.harvard.edu / abs / 2017arXiv170807471J},
         adsnote = {Provided by the SAO / NASA Astrophysics Data System}
         }
```
