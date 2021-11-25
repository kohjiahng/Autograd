from autograd.backfuncs.backfunc import BackFunc
class SigmoidBack(BackFunc):
    def __call__(self, tensor):
        A, = tensor._parents
        if A.requires_grad:
            A._add_grad(tensor.asarray() * (1 - tensor.asarray()))