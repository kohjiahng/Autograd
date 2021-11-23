from backfuncs.backfunc import BackFunc
import numpy as np
class TransposeBack(BackFunc):
    def __init__(self, axes = None):
        self.axes = axes
        self.inverse_axes = np.argsort(axes) if axes else None
    def __call__(self,tensor):
        assert len(tensor._parents) == 1, f"TransposeBack called with > 1 parents"
        A, = tensor._parents
        A._add_grad(tensor.grad.transpose(self.inverse_axes))