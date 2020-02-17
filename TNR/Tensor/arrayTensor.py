import numpy as np
from TNR.Tensor.tensor import Tensor
from TNR.Utilities.arrays import permuteIndices


class ArrayTensor(Tensor):

    def __init__(self, tens, logScalar=0):
        assert np.sum(np.isnan(tens)) == 0

        self._shape = tens.shape
        self._rank = len(self._shape)

        self._size = tens.size

        # We normalize the Tensor by factoring out the log of the
        # maximum-magnitude element.
        m = np.max(np.abs(tens))
        if m == 0:
            m = 1.
        self._logScalar = np.log(m) + logScalar
        self._array = np.copy(tens / m)

    def __str__(self):
        return 'Tensor of shape ' + str(self.shape) + '.'

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return self._rank

    @property
    def size(self):
        return self._size

    @property
    def compressedSize(self):
        return self._size

    @property
    def logScalar(self):
        return self._logScalar

    @property
    def array(self):
        return self._array * np.exp(self.logScalar)

    @property
    def scaledArray(self):
        return np.copy(self._array)

    def divideLog(self, log):
        '''        
        :param log: The logarithm of the factor by which to divide.
        :return: A new Tensor given by dividing this one by the exponential of the log.
        '''
        return self.multiplyLog(-log)

    def multiplyLog(self, log):
        '''        
        :param log: The logarithm of the factor by which to multiply.
        :return: A new Tensor given by multiplying this one by the exponential of the log.
        '''
        return ArrayTensor(self.scaledArray, logScalar=self.logScalar + log)

    def contract(self, ind, other, otherInd):
        '''
        Takes as input:
                ind 		-	A list of indices on this Tensor.
                other 		-	The other Tensor.
                otherInd	-	A list of indices on the other Tensor.

        Returns a Tensor containing the contraction of this Tensor with the other.
        '''
        if hasattr(other, 'network'):
            # If the other Tensor is a TreeTensor then it should handle the contraction.
            # The argument front tells the TreeTensor contract method to flip the
            # order of the contraction so the indices are in the order we
            # expect.
            return other.contract(otherInd, self, ind, front=False)
        else:
            arr = np.tensordot(
                self.scaledArray, other.scaledArray, axes=(
                    (ind, otherInd)))
            return ArrayTensor(arr, logScalar=self.logScalar + other.logScalar)

    def getIndexFactor(self, ind):
        return self.scaledArray, ind

    def setIndexFactor(self, ind, arr):
        return ArrayTensor(arr, logScalar=self.logScalar)

    def __deepcopy__(self, memo):
        return self
