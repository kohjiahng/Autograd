class SGD:
    def __init__(self, parameters, lr = 0.01):
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr