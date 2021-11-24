import numpy as np
import logging
from autograd.backfuncs.backfunc import BackFunc
class MatMulBack(BackFunc): # Need to add broadcasting
    def __call__(self,tensor):
        assert len(tensor._parents) == 2, f"MatMulBack called with {len(tensor._parents)} parents"
        A, B = tensor._parents
        A_arr, B_arr = A, B
        if type(A) is type(tensor):
            A_arr = A.asarray()
        if type(B) is type(tensor):
            B_arr = B.asarray()

        # Append/prepend 1 dim to 1d arrays
        if A_arr.ndim == 1:
            A_arr = A_arr[np.newaxis,...] # makes a copy
        if B_arr.ndim == 1:
            B_arr = B_arr[...,np.newaxis]

        A_shape = np.array([*[1] * (B_arr.ndim - A_arr.ndim), *A_arr.shape])
        B_shape = np.array([*[1] * (A_arr.ndim - B_arr.ndim), *B_arr.shape])

        par_grad = tensor.grad
        if A.ndim == 1:
            par_grad = par_grad[...,np.newaxis,:]
        elif B.ndim == 1:
            par_grad = par_grad[...,np.newaxis]


        if type(A) is type(tensor) and A.requires_grad:
            A_grad = (par_grad @ B_arr.swapaxes(-1, -2)) # Broadcasted
            A_grad = A_grad.sum(axis = tuple((A_shape[:-2] < B_shape[:-2]).nonzero()[0]), keepdims = True) # Broadcasted with random size 1 dims in front
            A_grad = A_grad.squeeze(tuple(range((B_arr.ndim - A_arr.ndim)))) # Prepended/appended 1s
            if A.ndim == 1:
                A_grad = A_grad.sum(axis = 0)
            A._add_grad(A_grad)

        if type(B) is type(tensor) and B.requires_grad:
            B_grad = (A_arr.swapaxes(-1, -2) @ par_grad) # Broadcasted
            B_grad = B_grad.sum(axis = tuple((B_shape[:-2] < A_shape[:-2]).nonzero()[0]), keepdims = True) # Broadcasted with random size 1 dims in front
            B_grad = B_grad.squeeze(tuple(range((A_arr.ndim - B_arr.ndim)))) # Prepended/appended 1s
            if B.ndim == 1:
                B_grad = B_grad.sum(axis = 1)
            B._add_grad(B_grad)