from backfuncs.backfunc import BackFunc
from copy import deepcopy

class ReciprocalBack(BackFunc):
    def __init__(self, numerator = 1):
        self.numerator = numerator
    def __call__(self, tensor):
        assert len(tensor._parents) == 1, f"ReciprocalBack called with > 1 parents"
        A, = tensor._parents
        A._add_grad(-self.numerator * tensor.grad / (A.asarray() ** 2))
        