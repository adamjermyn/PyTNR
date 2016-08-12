from bisect import bisect_left

'''
This class is currently the bottleneck.
More specifically, list insertions and removals
called by this class account for 75% of the total runtime.
Clearly optimization is called for...
'''

class PriorityList:
	def __init__(self):
		self.list = []
		self.vals = []

	def add(self, item, val):
		ind = bisect_left(self.vals, val)

		self.list.insert(ind, item)
		self.vals.insert(ind, val)

	def remove(self, item):
		ind = self.list.index(item)

		self.list.remove(item)

		del self.vals[ind]

