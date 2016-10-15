

class Link:
	def __init__(self, b1, b2):
		self.bucket1 = b1
		self.bucket2 = b2

	def otherBucket(self, bucket):
		if bucket == self.bucket1:
			return self.bucket2
		elif bucket == self.bucket2:
			return self.bucket1
		else:
			raise ValueError