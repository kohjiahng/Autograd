from backfuncs.backfunc import BackFunc
import numpy as np
class TransposeBack(BackFunc):
    def __init__(self, axes = None):
        self.axes = axes
        self.inverse_axes = np.argsort(axes) if axes else None
    def __call__(self,tensor):
        tensor._parents[0]._add_grad(tensor.grad.transpose(axes = self.inverse_axes))