from backfuncs.backfunc import BackFunc
from copy import deepcopy
from numbers import Number
import numpy as np
class MulBack(BackFunc):
    def __call__(self, tensor):

        assert len(tensor._parents) == 2, f"MulBack called with {len(tensor._parents)} parents, expected 2"
        A, B = tensor._parents

        if isinstance(A, Number):
            B._add_grad(A * tensor.grad)
            return
        if isinstance(B, Number):
            A._add_grad(B * tensor.grad)
            return
        # Prepending 1s to the shapes so they are of equal length
        A_shape = np.array([*[1] * (B.ndim - A.ndim), *A.shape])
        B_shape = np.array([*[1] * (A.ndim - B.ndim), *B.shape])

        # np.sum over the smaller axes to get gradients
        if type(A) is type(tensor) and A.requires_grad:
            A_grad = (B.asarray() * tensor.grad).sum(axis = tuple((A_shape < B_shape).nonzero()[0]), keepdims = True);
            A._add_grad(A_grad.squeeze(tuple(range((B.ndim - A.ndim)))))
        if type(B) is type(tensor) and B.requires_grad:
            B_grad = (A.asarray() * tensor.grad).sum(axis = tuple((B_shape < A_shape).nonzero()[0]), keepdims = True);
            B._add_grad(B_grad.squeeze(tuple(range((A.ndim - B.ndim)))))