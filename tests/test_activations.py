import sys
sys.path.insert(1, './')
from autograd import *
from copy import deepcopy
import numpy as np
import unittest
import logging_settings
import logging
import math
from test_utils import CustomAssertMixin

np.random.seed(1)

class TestTensorFunctions(unittest.TestCase, CustomAssertMixin):
    def test_sigmoid(self):
        A = tensor.rand((10,), requires_grad=True)
        tensor.Tensor.sigmoid(A).sum().backward()
        def operations(A):
            return tensor.Tensor.sigmoid(A).sum()

        self.assertGradient(operations, A.no_grad(), A.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)   

    def test_softmax(self):
        A = tensor.rand((10,), requires_grad=True)
        self.assertAlmostEqual(tensor.Tensor.softmax(A.no_grad()).sum().item(), 1)
        tensor.Tensor.softmax(A).l2().backward()
        def operations(A):
            return tensor.Tensor.softmax(A).l2()

        self.assertGradient(operations, A.no_grad(), A.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)   


if __name__ == '__main__':
    unittest.main()