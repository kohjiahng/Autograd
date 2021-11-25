import sys
sys.path.insert(1, './')
from autograd.losses import CrossEntropyLoss,MSELoss

from autograd import *
from copy import deepcopy
import numpy as np
import unittest
import logging_settings
import logging
import math
from test_utils import CustomAssertMixin

np.random.seed(1)

class TestActivations(unittest.TestCase, CustomAssertMixin):
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

    def test_softmax_2(self):
        A = tensor.rand((100,10,), requires_grad=True)
        (tensor.Tensor.softmax(A)*tensor.Tensor.softmax(A)).sum().backward()
        def operations(A):
            return (tensor.Tensor.softmax(A)*tensor.Tensor.softmax(A)).sum()

        self.assertGradient(operations, A.no_grad(), A.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)   

    def test_crossentropy(self):
        A = tensor.rand((1,5), requires_grad=True) * 100
        B = tensor.rand((1,5), requires_grad=True) * 100
        crossentropy = CrossEntropyLoss()
        crossentropy(A, B).backward()
        def operations(A=A.no_grad(),B=B.no_grad()):
            return crossentropy(A, B)

        self.assertGradient(lambda A: operations(A=A.no_grad()), A.no_grad(), A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B.no_grad(), B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)   
if __name__ == '__main__':
    unittest.main()