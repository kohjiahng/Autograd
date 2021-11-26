class Module:
    def __init__(self):
        pass
    def __call__(self,x):
        return self.forward(x)
    def parameters(self):
        return []
    def eval(self): #Eval mode
        for parameter in self.parameters():
            parameter.no_grad(inplace=True)
    def train(self): #Train mode
        for parameter in self.parameters():
            parameter.require_grad(inplace=True)