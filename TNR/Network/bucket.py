import itertools


class Bucket:
    '''
    A Bucket is a means of externally referencing an index of a tensor which handles the way in which
    the tensor is linked to other tensors.

    Each Bucket references exactly one index and up to one link, but a Bucket may be associated with
    multiple Nodes (and hence Tensors).
    '''
    
    newid = itertools.count().__next__

    def __init__(self, id=None):
        if id is None:
            self.id = Bucket.newid()
        else:
            self.id = id
        self.node = None
        self._link = None
        self.otherBucket = None
        self.linked = False

    @property
    def otherNode(self):
        return self.otherBucket.node

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, value):
        self._link = value
        if value is None:
            self.otherBucket = None
            self.linked = False
        else:
            self.otherBucket = value.otherBucket(self)
            self.linked = True
        
        self._link = value

    @property
    def index(self):
        return self.node.bucketIndex(self)

    @property
    def size(self):
        return self.node.tensor.shape[self.index]

    @property
    def otherSize(self):
        return self.otherNode.tensor.shape[self.otherBucket.index]
