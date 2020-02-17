import itertools


class Link:
    '''
    A Link is a means of indicating an intent to contract two tensors.
    This object has two Buckets, one for each Node being connected.
    '''
    
    newid = itertools.count().__next__

    def __init__(self, b1, b2):
        assert not b1.linked
        assert not b2.linked
        self.id = Link.newid()
        self.bucket1 = b1
        self.bucket2 = b2
        b1.link = self
        b2.link = self

    def otherBucket(self, bucket):
        if bucket == self.bucket1:
            return self.bucket2
        elif bucket == self.bucket2:
            return self.bucket1
        else:
            raise ValueError
