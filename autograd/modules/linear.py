from autograd.modules.module import Module
import numpy as np
import autograd.tensor as tensor
class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int, bias = True, init_weights = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_weights = init_weights
        if not self.init_weights:
            self.init_weights = lambda x: np.random.normal(size = x)

        self.weights = tensor.tensor(self.init_weights((input_dim, output_dim)), requires_grad=True)

        if bias:
            self.bias = tensor.tensor(self.init_weights((output_dim,)), requires_grad=True)
        else:
            self.bias = tensor.zeros(output_dim, requires_grad = False)

    def forward(self, X):
        # Inputs are of shape (N, F)

        assert X.ndim == 2, f"Linear Expected 2d array, found {X.ndim}d array"
        assert X.shape[1] == self.input_dim, f"Number of features in X do not match input_dim ({X.shape[1]} vs {self.input_dim})"
        return X @ self.weights + self.bias
    def parameters(self):
        return [self.weights, self.bias]