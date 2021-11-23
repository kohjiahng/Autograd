from backfuncs.backfunc import BackFunc
class NegBack(BackFunc):
    def __call__(self, tensor):
        assert len(tensor._parents) == 1, "NegBack called with > 1 parent"
        A, = tensor._parents
        A._add_grad(-tensor.grad)