

################################
# Miscillaneous Helper Functions
################################

def tupleReplace(tpl, i, j):
	'''
	Returns a tuple with element i of tpl replaced with the quantity j.
	If j is None, just removes element i.
	'''
	assert i >= 0
	assert i < len(tpl)

	tpl = list(tpl)
	if j is not None:
		tpl = tpl[:i] + [j] + tpl[i+1:]
	else:
		tpl = tpl[:i] + tpl[i+1:]
	return tuple(tpl)