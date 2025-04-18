from autograd.tensor import Tensor
from autograd.operation import *


def exp(x: Tensor | float | int) -> Tensor | float | int:
    return np.e ** x

def sigmoid(x: Tensor | float | int) -> Tensor | float | int:
    return 1 / (1 + exp(-x))

def relu(x: Tensor | float | int) -> Tensor:
    class Relu(BaseOperation):
        def _forward(self):
            return self._get_arg(0) * np.where(self._get_arg(0) > 0, 1, 0)
    
        def _backward(self, i):
            return np.where(self._get_arg(i) > 0, 1, 0)
        
        def info(self, verbose=False):
            return f"ReLU ({self._join_args(verbose=verbose)})"
    
    op = Relu()
    return op.forward(x)

def tanh(x: Tensor | float | int) -> Tensor:
    pass

def softmax(x: Tensor) -> Tensor:
    pass

def mean_squared_error(x: Tensor | float | int) -> float:
    pass

def cross_entropy(x: Tensor) -> float:
    pass
