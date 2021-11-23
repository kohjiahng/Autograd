from backfuncs import *
import tensor
from copy import deepcopy
import numpy as np
import unittest
import logging_settings
import logging
import math
np.random.seed(0)
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
    def assertArrayAlmostEqual(self, A, B, msg = None, delta = 1e-2):
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
        
        self.assertArrayAlmostEqual(numericGradient(function, X, epsilon=1e-6) , grad)

class TestTensorFunctions(unittest.TestCase, CustomAssertMixin):
    def assertArrayEqual(self, A, B, msg = None):
        self.assertTrue(
            np.array_equal(A, B),
            msg if msg else f"Array {A} != {B}"
        )

    def assertArrayAlmostEqual(self, A, B, msg = None, delta = 1e-3):
        self.assertAlmostEqual(
            np.absolute(A - B).max(), 
            0,
            msg = msg if msg else f"Array {A} !~ {B}",
            delta = delta
        )
    def test_add(self):
        A = tensor.tensor([1,2])
        B = tensor.tensor([2,3])
        C = tensor.tensor([3,5])

        self.assertArrayEqual(A + B, C)      
    def test_sub(self):
        A = tensor.tensor([1,2])
        B = tensor.tensor([2,3])
        C = tensor.tensor([-1,-1])

        self.assertArrayEqual(A - B, C)
    def test_mul(self):
        A = tensor.tensor([1,2])
        B = tensor.tensor([2,3])
        C = tensor.tensor([2,6])
        self.assertArrayEqual(A * B, C)
    def test_div(self):
        A = tensor.tensor([1,2])
        B = tensor.tensor([2,3])
        C = tensor.tensor([0.5,2 / 3])
        self.assertArrayEqual(A / B, C)
    def test_neg(self):
        A = tensor.tensor([[1., 2], [2, 3]])
        B = tensor.tensor([[-1., -2], [-2, -3]])
        self.assertArrayEqual(-A, B)
    def test_matmul_1(self): # matrix @ vector
        A = tensor.tensor([[1.,2,3], [2,3,4], [3,4,5], [1,3,4]])
        B = tensor.tensor([1.,2,3])
        C = tensor.tensor([14., 20, 26, 19])
        self.assertArrayEqual(A @ B, C)
    def test_matmul_2(self): # matrix @ matrix
        A = tensor.tensor([[1., 2], [2, 3], [3,4]]) # (3, 2)
        B = tensor.tensor([[1., 2, 3], [2, 3, 4]]) # (2, 3)
        C = tensor.tensor([[5., 8, 11], [8, 13, 18],[11, 18, 25]])
        self.assertArrayEqual(A @ B, C)
    def test_sum(self):
        A = tensor.tensor([[1., 2], [2, 3]])
        B = tensor.tensor(8.)
        self.assertArrayEqual(A.sum(), B)

class TestTensorBackwardOperators(unittest.TestCase, CustomAssertMixin):
    def test_add(self):
        A = tensor.tensor(1., requires_grad=True)
        B = tensor.tensor(2., requires_grad=True)
        C = 2 * (A + B)
        C.backward()

        A_grad = np.array(2.)
        B_grad = np.array(2.)

        self.assertArrayEqual(A.grad, A_grad)
        self.assertArrayEqual(B.grad, B_grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_add_2(self): # Test broadcasting
        A = tensor.tensor([
            [1, 2, 3],
            [1, 3, 2],
            [2, 2, 1]
        ], requires_grad=True)
        B = tensor.tensor([2,3,2], requires_grad=True)
        C = A + B
        C.sum().backward()

        A_grad = np.array([
            [1,1,1],
            [1,1,1],
            [1,1,1]
        ])
        B_grad = np.array([3,3,3])
        self.assertArrayEqual(A.grad, A_grad)
        self.assertArrayEqual(B.grad, B_grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_mul(self):
        A = tensor.tensor(1., requires_grad=True)
        B = tensor.tensor(2., requires_grad=True)
        C = A * B
        C.backward()

        A_grad = np.array(2.)
        B_grad = np.array(1.)

        self.assertArrayEqual(A.grad, A_grad)
        self.assertArrayEqual(B.grad, B_grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_mul_2(self): # Test broadcasting
        A = tensor.rand((3,3), requires_grad=True)
        B = tensor.rand((3,), requires_grad=True)
        (A*B).sum().backward()
        def operations(A=A.no_grad(), B=B.no_grad()):
            C = (A * B)
            return C.sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A, A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)      
    def test_rmul(self): # Test broadcasting
        A = tensor.rand((3,3), requires_grad=True)
        B = np.random.rand(*(3,))
        (A*B).sum().backward()
        def operations(A=A.no_grad(), B=B):
            C = (A * B)
            return C.sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A, A.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)      
    def test_matmul(self):
        A = tensor.rand((3,3,5), requires_grad=True)
        B = tensor.rand((5,), requires_grad = True)
        (A@B).sum().backward()

        def operations(A, B):
            C = (A @ B)
            return C.sum()
        self.assertGradient(lambda A: operations(A=A.no_grad(), B=B.no_grad()), A, A.grad)
        self.assertGradient(lambda B: operations(A=A.no_grad(), B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_matmul_2(self):
        A = tensor.rand((5,), requires_grad=True)
        B = tensor.rand((3,5,3), requires_grad = True)
        (A@B).sum().backward()

        def operations(A=A.no_grad(), B=B.no_grad()):
            C = (A @ B)
            return C.sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A, A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_matmul_3(self):
        A = tensor.rand((5,4,5), requires_grad=True)
        B = tensor.rand((1,5,3), requires_grad = True)
        (A@B).sum().backward()

        def operations(A=A.no_grad(), B=B.no_grad()):
            C = (A @ B)
            return C.sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A, A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_matmul_4(self):
        A = tensor.rand((5,4,5), requires_grad=True)
        B = np.random.rand(*(1,5,3))
        (A@B).sum().backward()

        def operations(A=A.no_grad(), B=B):
            C = (A @ B)
            return C.sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A, A.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_rmatmul(self):
        A = np.random.rand(*(5,4,5))
        B = tensor.rand((1,5,3), requires_grad=True)
        (A@B).sum().backward()

        def operations(A=A, B=B.no_grad()):
            C = (A @ B)
            return C.sum()

        self.assertGradient(lambda B: operations(B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_truediv(self):
        A = tensor.tensor(1., requires_grad=True)
        B = tensor.tensor(2., requires_grad=True)
        C = A / B
        C.backward()

        A_grad = np.array(0.5)
        B_grad = np.array(-0.25)

        self.assertArrayEqual(A.grad, A_grad)
        self.assertArrayEqual(B.grad, B_grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_truediv_2(self): # Test broadcasting
        A = tensor.rand((3,3), requires_grad=True)
        B = tensor.rand((3,), requires_grad=True)
        (A/B).sum().backward()
        def operations(A=A.no_grad(), B=B.no_grad()):
            return (A / B).sum()

        self.assertGradient(lambda A: operations(A=A.no_grad()), A, A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)   

    def test_rtruediv(self): # Test broadcasting
        A = np.random.rand(*(3,3))
        B = tensor.rand((3,), requires_grad=True)
        (A/B).sum().backward()
        def operations(A=A, B=B.no_grad()):
            return (A / B).sum()

        self.assertGradient(lambda B: operations(B=B.no_grad()), B, B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)   
class TestTensorBackwardFunctions(unittest.TestCase, CustomAssertMixin):
    def test_sum(self):
        A = tensor.rand((2, 2, 2, 2), requires_grad=True)
        B = tensor.rand((2, 2), requires_grad=True)

        # A = tensor.rand((2,2),requires_grad=True)
        # B = tensor.tensor([1.,2],requires_grad=True)
        (A.sum(axis = 0) * B).sum().backward()

        def operations(A=A.no_grad(), B=B.no_grad()):
            return (A.sum(axis = 0) * B).sum()

        self.assertGradient(lambda A: operations(A=A.no_grad()), A.copy(), A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B.copy(), B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_transpose(self):
        A = tensor.rand((2, 3, 2), requires_grad=True)
        B = tensor.rand((2, 2, 3), requires_grad=True)
        (A.transpose((0, 2, 1)) * B).sum().backward()
        def operations(A=A.no_grad(), B=B.no_grad()):
            return (A.transpose((0, 2, 1)) * B).sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A.copy(), A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B.copy(), B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)
    def test_flatten(self):
        A = tensor.rand((2, 3, 2), requires_grad=True)
        B = tensor.rand((12,), requires_grad=True)
        (A.flatten() * B).sum().backward()
        def operations(A=A.no_grad(), B=B.no_grad()):
            return (A.flatten() * B).sum()
        self.assertGradient(lambda A: operations(A=A.no_grad()), A.copy(), A.grad)
        self.assertGradient(lambda B: operations(B=B.no_grad()), B.copy(), B.grad)
        self.assertEqual(tensor.Tensor.total_connections, 0)

        

if __name__ == '__main__':
    unittest.main()