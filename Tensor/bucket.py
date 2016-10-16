
class Bucket:

	def __init__(self):
		self.node = None
		self.link = None

	@property
	def otherBucket(self):
		return self.link.otherBucket(self)