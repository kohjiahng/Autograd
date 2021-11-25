from autograd.backfuncs.backfunc import BackFunc
class ScalarMaxBack(BackFunc):
    def __init__(self, c):
        self.c = c
    def __call__(self,tensor):
        assert len(tensor._parents) == 1, f"ScalarMaxBack called with > 1 parents"
        A, = tensor._parents
        grad_arr = tensor.grad.copy()
        grad_arr[A == 0] = 0
        A._add_grad(grad_arr)