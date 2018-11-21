import numpy as np
from TNR.Tensor.arrayTensor import ArrayTensor
from TNR.TreeTensor.treeTensor import TreeTensor
from TNR.TreeTensor.identityTensor import IdentityTensor

def binary_switch(accuracy, data_dimension_1, data_dimension_2):
	'''
	Constructs a binary switch tensor S_{ijkl}.
	The first index is the binary input which controls the switch.
	The second index is the output data line.
	The final two indices are the input data lines.
	When i == 0 the first input data line is copied to the output data line.
	When i == 1 the second input data line is copied to the output data line.
	The input data lines have dimensions given by data_dimension_1 and data_dimension_2.
	The output data line has dimension given by the larger of these.
	'''

	out_data_dimension = max(data_dimension_1, data_dimension_2)

	switch = np.zeros((2, out_data_dimension, data_dimension_1, data_dimension_2))
	switch[0] = np.identity(out_data_dimension)[:,:data_dimension_1,np.newaxis]
	switch[1] = np.identity(out_data_dimension)[:,np.newaxis,:data_dimension_2]
	switch_tensor = TreeTensor(accuracy)
	switch_tensor.addTensor(ArrayTensor(switch))

	return switch_tensor

def multi_bit_switch(accuracy, data_dimension_1, data_dimension_2, num_bits):
	'''
	Constructs a binary switch tensor which takes as input num_bits bits and
	switches on the last of these. The remaining bits are copied and passed through 
	to two different channels corresponding to the two input data lines.
	The result is of the form:

	S_{(b1, b2, ..., bN), out_data, (b1, b2, ..., b(N-1)), in_data_1, (b1, b2, ..., b(N-1)), in_data_2}
	'''

	assert num_bits > 0

	switch_tensor = binary_switch(accuracy, data_dimension_1, data_dimension_2)
	bitCopiers = list(IdentityTensor(2, 3, accuracy) for _ in range(num_bits-1))

	for b in bitCopiers:
		switch_tensor = switch_tensor.contract([], b, [])

	# Now switch_tensor has indices ordered as:
	# bN, out_data, in_data_1, in_data_2, b1, b1, b1, b2, b2, b2, ...
	# So we rearrange these into the form described above.

	# Get the input set of bits
	new_ext_buckets = []
	for i in range(num_bits-1):
		new_ext_buckets.append(switch_tensor.externalBuckets[4+3*i])
	new_ext_buckets.append(switch_tensor.externalBuckets[0])

	# Output data
	new_ext_buckets.append(switch_tensor.externalBuckets[1])

	# First output bit channel
	for i in range(num_bits-1):
		new_ext_buckets.append(switch_tensor.externalBuckets[5+3*i])

	# First input data channel
	new_ext_buckets.append(switch_tensor.externalBuckets[2])

	# Second output bit channel
	for i in range(num_bits-1):
		new_ext_buckets.append(switch_tensor.externalBuckets[6+3*i])

	# Second input data channel
	new_ext_buckets.append(switch_tensor.externalBuckets[3])

	switch_tensor.externalBuckets = new_ext_buckets

	return switch_tensor

def switch_tree(accuracy, data_dimensions):
	'''
	Constructs a tree of switches of the form:

	S_{b1,b2,...,bN,in_data_1,in_data_2,...,in_data_len(data_dimensions)}

	where N is the least integer such that 2^N > len(data_dimensions).
	When this object is dotted against a tensor with len(data_dimensions) external indices
	the result is a tensor containing all partial traces that leave a single index intact.
	The partial trace over the k-th index (i.e. the one attached to the k-th data input 
	of the switch tree) may be accessed by setting b1,b2,...,bN to the binary string equalling
	k.

	If len(data_dimensions) < 2^N this tensor is zero for all k > len(data_dimensions).

	'''

	# Find tree depth
	N = np.log2(len(data_dimensions))
	N = int(N)
	if 2**N < len(data_dimensions):
		N += 1

	# Pad data_dimensions out to 2**N
	padding = 2**N - len(data_dimensions)
	data_dimensions = list(data_dimensions) + list(1 for _ in range(padding))

	# Construct root
	data_dimension_1 = max(data_dimensions[:len(data_dimensions)//2])
	data_dimension_2 = max(data_dimensions[len(data_dimensions)//2:])
	root = multi_bit_switch(accuracy, data_dimension_1, data_dimension_2, N)

	# Construct layers
	for i in range(1,N):
		row = []
		stride = len(data_dimensions)//2**(i+1)
		for j in range(2**i):
			data_dimension_1 = max(data_dimensions[stride*(2*j):stride*(2*j+1)])
			data_dimension_2 = max(data_dimensions[stride*(2*j+1):stride*(2*j+2)])
			new = multi_bit_switch(accuracy, data_dimension_1, data_dimension_2, N-i)
			row.append(new)
		start = N + 1
		for j in range(2**i):
			num_bits = N - i
			num_out = num_bits + 1
			num_in = 2 * (num_out - 1)
			root = root.contract(range(start, start + num_out), row[j], range(num_out), elimLoops=False)

			root.externalBuckets = root.externalBuckets[:start] + root.externalBuckets[-num_in:] + root.externalBuckets[start:-num_in]
			start += num_in

	# Dot all padded readouts with [1.]
	for i in range(padding):
		root = root.contract([root.rank - 1], ArrayTensor(np.ones((1,))), [0], elimLoops=False)


	root.network.contractRank2()

	root.externalBuckets = root.externalBuckets[:N][::-1] + root.externalBuckets[N::]

	root.eliminateLoops()

	return root, N





