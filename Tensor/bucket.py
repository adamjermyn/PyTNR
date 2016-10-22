class Bucket:

	def __init__(self):
		self.node = None
		self.link = None

	@property
	def otherBucket(self):
		return self.link.otherBucket(self)

	@property
	def otherNode(self):
		return self.link.otherBucket(self).node

	@property
	def linked(self):
		return (self.link is not None)

	@property
	def index(self):
		return self.node.bucketIndex(self)