from autograd.backfuncs.backfunc import BackFunc
class ReshapeBack(BackFunc):
    def __init__(self, order):
        self.order = order
    def __call__(self,tensor):
        assert len(tensor._parents) == 1, f"ReshapeBack called with > 1 parents"
        A, = tensor._parents
        A._add_grad(tensor.grad.reshape(A.shape,order=self.order))