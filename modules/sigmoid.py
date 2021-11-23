from modules.module import Module
import tensor
class Sigmoid(Module):
    def __init__(self):
        pass

    def forward(self, X):
        return tensor.Tensor.sigmoid(X)