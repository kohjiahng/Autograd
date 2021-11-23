from backfuncs.backfunc import BackFunc
class SigmoidBack(BackFunc):
    def __call__(self, tensor):
        if tensor._parents[0].requires_grad:
            tensor._parents[0]._add_grad(tensor.asarray() * (1 - tensor.asarray()))