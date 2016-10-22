import itertools

class Bucket:

	newid = itertools.count().next

	def __init__(self):
		self.node = None
		self.link = None
		self.id = Bucket.newid()

	def __hash__(self):
		return self.id

	@property
	def otherBucket(self):
		return self.link.otherBucket(self)

	@property
	def otherNode(self):
		return self.link.otherBucket(self).node

	@property
	def linked(self):
		return (self.link is not None)
