
class Bucket:

	def __init__(self, node):
		self.node = node
		self.link = None

	@property
	def otherBucket(self):
		return self.link.otherBucket(self)