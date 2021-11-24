from autograd.modules.module import Module
import numpy as np
import autograd.tensor as tensor
class Flatten(Module):
    def forward(self, X):
        # Inputs are of shape (N, ...)
        return X.reshape((len(X), -1))
    