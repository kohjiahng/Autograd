import numpy as np
from copy import deepcopy
import logging
from numbers import Number
from backfuncs.reciprocalback import ReciprocalBack
import logging_settings
from backfuncs import *

class Tensor(np.ndarray):
    total_connections = 0 # Total number of operators in computation graph

    def __new__(cls, arr, *args, **kwargs):
        dtype = None
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        elif kwargs.get('requires_grad'):
            dtype = 'float64'

        obj = np.asarray(arr, dtype=dtype).view(cls)
        return obj
    def __init__(self, arr, requires_grad = False, _parents = (), _back = None, *args, **kwargs):
        '''
        _back is a function which takes in an object and propogates gradients to parents
        '''
        self.requires_grad = requires_grad
        self._parents = _parents
        if requires_grad:
            self._back = _back
            

            if isinstance(_back, BackFunc):
                Tensor.total_connections += 1

            self.grad = np.zeros(self.shape, dtype=self.dtype)
            self._out_degree, self._resolved_degree = 0, 0 # Out degree in computation graph

            for parent in self._grad_parents():
                parent._out_degree += 1
        else:
            self._back, self.grad, self._out_degree, self._resolved_degree = None, None, None, None

    def asarray(self):
        return self.__array__()
    
    def _add_grad(self, x):
        if not self.requires_grad:
            return
        self.grad += x
    def _sub_grad(self, x):
        if not self.requires_grad:
            return
        self.grad -= x
    def _mul_grad(self, x):
        if not self.requires_grad:
            return
        self.grad *= x 
    def _div_grad(self, x):
        if not self.requires_grad:
            return
        self.grad /= x
    def _remove_back(self):
        if self._back:
            Tensor.total_connections -= 1
        self._back = None
    
    def _grad_parents(self):
        for parent in self._parents:
            if isinstance(parent, Tensor) and parent.requires_grad:
                yield parent
    def __add__(self, other):
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            return Tensor(self.asarray() + asarray(other), _parents = (self, other), _back = AddBack(), requires_grad = True)
        else:
            return Tensor(self.asarray() + asarray(other), requires_grad = False)

    
    def __mul__(self, other):
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            return Tensor(self.asarray() * asarray(other), _parents = (self, other), _back = MulBack(), requires_grad = True)
        else:
            return Tensor(self.asarray() * asarray(other), requires_grad = False)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (1 / other)
    
    def __neg__(self):
        if self.requires_grad:
            return Tensor(-self.asarray(), _parents = (self,), _back = NegBack(), requires_grad = True)
        else:
            return Tensor(-self.asarray(), requires_grad = False)
    def __matmul__(self, other):
        assert isinstance(other, np.ndarray), "__matmul__ only supported for np.ndarray subclasses"
        if self.ndim == 1 and other.ndim == 1:
            return self.dot(other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            return Tensor(self.asarray() @ asarray(other), _parents = (self,other), _back = MatMulBack(), requires_grad = True)
        else:
            return Tensor(self.asarray() @ asarray(other), requires_grad = False)
    
    def __radd__(self, other):
        return self.__add__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __rsub__(self, other):
        return self.__mul__(other)
    def __rtruediv__(self, other):
        if isinstance(other, Number):
            if self.requires_grad:
                return Tensor(other / self.asarray(), _parents = (self,), _back = ReciprocalBack(numerator=other), requires_grad = True)
            else:
                return Tensor(other / self.asarray(), requires_grad = False)
        else:
            return other * (1 / self)

    def __rmatmul__(self, other):
        assert isinstance(other, np.ndarray), f"__rmatmul__ only supported for np.ndarray subclasses, called for {type(other)}"
        if self.ndim == 1 and other.ndim == 1:
            return self.dot(other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            return Tensor(asarray(other) @ self.asarray() , _parents = (other,self), _back = MatMulBack(), requires_grad = True)
        else:
            return Tensor(asarray(other) @ self.asarray() , requires_grad = False)
    def sum(self, axis = None):
        if self.requires_grad:
            return Tensor(
                super().sum(axis=axis),
                _parents = (self,), _back = SumBack(axis), requires_grad = True
            )
        else:
            return Tensor(super().sum(axis=axis), requires_grad = False)
    def transpose(self, axes = None):
        if self.requires_grad:
            return Tensor(
                super().transpose(axes),
                _parents = (self, ), _back = TransposeBack(axes), requires_grad=True
            )
        else:
            return Tensor(super().transpose(axes), requires_grad = False)
    def dot(self, other):
        return (self * other).sum()

    def flatten(self):
        if self.requires_grad:
            return Tensor(self.__array__.reshape(-1), _back = FlattenBack(), requires_grad = True)
        else:
            return Tensor(self.__array__.reshape(-1), requires_grad = False)

    @staticmethod
    def sigmoid(tensor):
        if tensor.requires_grad:
            return Tensor(1 / (1 - np.exp(-tensor.asarray())), requires_grad=True, _back=SigmoidBack())
        else:
            return Tensor(1 / (1 - np.exp(-tensor.asarray())), requires_grad=False)
    def _backward(self):
        if not self.requires_grad:
            return
        if self._parents:
            self._back(self)
            for parent in self._grad_parents():
                parent._resolved_degree += 1
                if parent._out_degree == parent._resolved_degree:
                    parent._backward()
    def no_grad(self, inplace = False):
        if inplace:
            self.grad, self._back, self._parents = None, None, ()
            self.requires_grad = False
            return self
        else:
            _new = deepcopy(self)
            _new.grad, _new._back, _new._parents = None, None, ()
            _new.requires_grad = False
            return _new
    def zero_grad(self):
        self.grad.fill(0)
        self._resolved_degree = 0

        for parent in self._grad_parents():
            parent.zero_grad()
    def clear_graph(self):
        if self.requires_grad:
            for parent in self._grad_parents():
                parent.clear_graph()
            self._parents = ()
            self._remove_back()
    def backward(self):
        assert self.requires_grad, "Array has to have requires_grad = True to call backward"
        assert self.size == 1, "Array has to be size 1 to call backward"
        self.zero_grad()
        self.grad.fill(1)
        self._backward()
        self.clear_graph() 

def asarray(tensor):
    return tensor.__array__() if isinstance(tensor, Tensor) else tensor

def tensor(arr,requires_grad = False, *args, **kwargs):
    return Tensor(arr, requires_grad = requires_grad, *args, **kwargs)

def zeros(shape,requires_grad=False,*args, **kwargs):
    return Tensor(np.zeros(shape), requires_grad=requires_grad*args, **kwargs)

def ones(shape,requires_grad=False, *args, **kwargs):
    return Tensor(np.ones(shape), requires_grad=requires_grad,*args, **kwargs)
def rand(shape,requires_grad=False, *args, **kwargs):
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad,*args, **kwargs)