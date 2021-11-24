from autograd.modules.module import Module
class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules
    def forward(self, X):
        for module in self.modules:
            X = module(X)
        return X
    def parameters(self):
        return [parameter for module in self.modules for parameter in module.parameters()]