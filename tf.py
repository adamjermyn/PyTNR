

'''
A Hamiltonian is represented as a list of tuples.
Each tuple contains a 2D numpy array specifying the offsets to the interaction terms
and a scalar specifying the interaction energy.

For instance, a nearest-neighbor Hamiltonian on a square lattice is:

H = [
	([],h),
	([[0,1]],J),
	([[1,0]],J),
	]

Note that we omit the negative offsets because then we would be double counting.


The Transfer Tensor is a tensor which exists at each site.
This tensor has an index for every location specified in the Hamiltonian,
as well as for the site it is located on.
The tensor is evaluated as:

T_{ijkl...} = Exp[-H(ijkl...)]

The partition function is then given by tracing the product of all
transfer tensors over all indices.
Note that some transfer tensors from different sites will share indices.
As a result this trace cannot be done simply over each tensor independently.

In general, this trace is prohibitively expensive.
As a result we are looking for an approximation to the trace of the transfer tensors.
In particular, our approach will be to search for singular value decompositions of the
full product of tensors.
To be specific, suppose we wish to trace out the index i.
The full tensor is

T_{i,other} = Product over tensors involving i * Product over other tensors
			 = W_{i,other}*E_{other,other'}

where other and other' have no indices in common and do not include i.
Tracing over i may be achieved trivially, but in doing so we increase the complexity
of the remaining tensors, as we have gone from many small tensors each
involving i and a few other indices to a single tensor involving a large number of indices.
To remedy this we use a singular value decomposition on each tensor involving i.
This gives

T_{i,other} = V_{i,j} lambda_{j} U_{j,other}

This trace may be evaluated 
'''

# Equal entropy division

'''
Represent tensor network as graph.
Pick bond with highest mutual entropy, merge. Pick merger SVD split based on putting half of the external
entropy on each side of the split. This ought to maximize reduction in entropy associated with tracing over
the inner index and producing an SVD split.
Iterate on this, processing in parallel by coloring the graph and performing mergers involving a single color
with any other color (making choices where applicable by maximizing bond entropy).

Store the graph at each iteration in a tree structure, such that it is easy to tell which merged tensor
corresponds to which unmerged children.
To account for environmental effects, isolate a child at the lowest level and SVD with the environment
incorporated based on the highest-level approximation possible.
Then go to a child which shares a parent with the original one and repeat.
Repeat for all such children, then update the parent SVD (ignoring environment).
Move on to a child sharing a grandparent with the original child, and repeat.
Once all lowest-level children have been optimized, move up a level and repeat.
Repeat until tensors at all levels have been optimized.
Repeat until converged.

The alternative (top-down) is not preferred because it requires doing non-environmental updates on
tensors which have previously been environmentally updated (as the environmental updates are expensive,
	and so shouldn't be iterated over and over).

'''
