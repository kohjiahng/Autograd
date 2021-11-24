class MSELoss:
    def __init__(self):
        pass
    def __call__(self, yhat, y):
        return (yhat - y).l2() / y.size