from backfuncs.backfunc import BackFunc
import numpy as np
class SumBack(BackFunc):
    def __init__(self, axis = None):
        self.axis = axis
    def __call__(self,tensor):
        assert len(tensor._parents) == 1, "SumBack called with > 1 parent"
        A, = tensor._parents
        if self.axis:
            A._add_grad(np.expand_dims(tensor.grad,self.axis)) # Broadcasting :)
        else:
            A._add_grad(tensor.grad)