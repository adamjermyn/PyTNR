import itertools


class Bucket:
    '''
    A Bucket is a means of externally referencing an index of a tensor which handles the way in which
    the tensor is linked to other tensors.

    Each Bucket references exactly one index and up to one link, but a Bucket may be associated with
    multiple Nodes (and hence Tensors).
    '''
    
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
