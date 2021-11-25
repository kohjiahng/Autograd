from autograd.backfuncs.backfunc import BackFunc
import numpy as np
import logging
class SoftmaxBack(BackFunc):
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, tensor):
        assert len(tensor._parents) == 1, "SoftmaxBack called with > 1 parent"
        A, = tensor._parents
        par_arr = tensor.asarray()
        A._add_grad(-par_arr * (par_arr * tensor.grad).sum(axis=self.axes,keepdims=True) + par_arr * tensor.grad) # Weird ass shit