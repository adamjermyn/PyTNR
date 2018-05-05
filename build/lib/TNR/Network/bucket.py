import itertools


class Bucket:
    newid = itertools.count().__next__

    def __init__(self):
        self.id = Bucket.newid()
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

    @property
    def size(self):
        return self.node.tensor.shape[self.index]

    @property
    def otherSize(self):
        return self.otherNode.tensor.shape[self.otherBucket.index]
