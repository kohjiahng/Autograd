import numpy as np
import autograd.tensor as tensor
class CrossEntropyLoss:
    def __init__(self):
        pass
    def __call__(self, yhat, y):
        # 1e-15 added to prevent log0 errors
        assert yhat.ndim == 2 and y.ndim == 2, f"Cross Entropy Loss called with {yhat.ndim} dims, expected 2"
        return -tensor.Tensor.log(yhat + 1e-15).dot(y)