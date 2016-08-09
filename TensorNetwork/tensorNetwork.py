





class TensorNetwork:
	def __init__(self):
		self.tensors = set()
		self.all_links = set()
		self.idDict = {}
		self.idCounter = 0