from tensor import Tensor
import numpy as np
from compress import compress

def mergeLinks(n1, n2, compressLink=False):
	assert n1 in n2.connectedHigh()
	assert n2 in n1.connectedHigh()
	assert n1 in n1.network().topLevelNodes()
	assert n2 in n2.network().topLevelNodes()

	links = n1.linksConnecting(n2)

	indices1 = []
	indices2 = []

	for link in links:
		if link.bucket1().topNode() == n1:
			indices1.append(n1.bucketIndex(link.bucket1()))
			indices2.append(n2.bucketIndex(link.bucket2()))
		else:
			indices1.append(n1.bucketIndex(link.bucket2()))
			indices2.append(n2.bucketIndex(link.bucket1()))

	perm1 = [i for i in range(len(n1.tensor().shape())) if i not in indices1]
	perm2 = [i for i in range(len(n2.tensor().shape())) if i not in indices2]

	m1 = min(indices1)
	m2 = min(indices2)

	perm1 = perm1[:m1] + indices1 + perm1[m1:]
	perm2 = perm2[:m2] + indices2 + perm2[m2:]

	arr1 = n1.tensor().array()
	arr2 = n2.tensor().array()

	arr1m = np.transpose(arr1, axes=perm1)
	arr2m = np.transpose(arr2, axes=perm2)

	arr1m = np.reshape(arr1m, list(arr1m.shape[:m1]) + [np.product([arr1.shape[i] for i in indices1])] + list(arr1m.shape[m1+len(indices1):]))
	arr2m = np.reshape(arr2m, list(arr2m.shape[:m2]) + [np.product([arr2.shape[i] for i in indices2])] + list(arr2m.shape[m2+len(indices2):]))

	# Now the new index is where m1/m2 were.

	t1m = Tensor(arr1m.shape, arr1m)
	t2m = Tensor(arr2m.shape, arr2m)

	# We can delete the buckets associated with the removed indices
	ind1m = list(indices1)
	ind1m.remove(m1)
	ind2m = list(indices2)
	ind2m.remove(m2)

	n1m = n1.modify(t1m, delBuckets=ind1m, repBuckets=[m1])
	n2m = n2.modify(t2m, delBuckets=ind2m, repBuckets=[m2])

	n1m.addLink(n2m, m1, m2, compressed = False, children=links)

	if compressLink:
		link = n1m.findLink(n2m)
		_, n1m, n2m = compress(link)

	return n1m, n2m
