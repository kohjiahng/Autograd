from backfuncs.backfunc import BackFunc
class NegBack(BackFunc):
    def __call__(self, tensor):
        for parent in tensor._parents:
            parent._sub_grad(tensor.grad)