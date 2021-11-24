class Module:
    def __init__(self):
        pass
    def __call__(self,x):
        return self.forward(x)
    def parameters(self):
        return []