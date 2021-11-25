from autograd import tensor
import math
import numpy as np
import logging
def numericGradient(function, X, epsilon: float = 1e-6):
    '''
    Function takes in 1 array argument X, outputs a single value
    '''
    it = np.nditer(X, flags=['multi_index','refs_ok'])
    fX = function(X)
    grad = np.empty(X.shape)
    for _ in it:
        Y = X.asarray().copy()
        Y[it.multi_index] += epsilon
        Y = tensor.tensor(Y)
        grad[it.multi_index] = ((function(Y) - fX) / epsilon).item()
    return grad

class CustomAssertMixin:
    def assertArrayEqual(self, A, B):
        assert np.array_equal(A, B), f"Array {A} != {B}"
    def assertArrayAlmostEqual(self, A, B, msg = None):
        for i,j in zip(A.reshape(-1), B.reshape(-1)):
            if max(abs(i), abs(j)) < 1:
                self.assertTrue(
                    math.isclose(i, j, abs_tol=1e-3), msg = f"{i} != {j} with absolute tolerance {1e-3}"
                )
            else:
                self.assertTrue(
                    math.isclose(i, j, rel_tol=1e-3), msg = f"{i} != {j} with relative tolerance {1e-3}"
                )
    def assertGradient(self, function, X, grad): # Uses numeric methods to get gradient
        self.assertArrayAlmostEqual(numericGradient(function, X) , grad)
