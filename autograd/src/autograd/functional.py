from autograd.tensor import Tensor
from autograd.operation import *


def exp(x: Tensor | float | int) -> Tensor | float | int:
    return np.e ** x

def sigmoid(x: Tensor | float | int) -> Tensor | float | int:
    return 1 / (1 + exp(-x))

def relu(x: Tensor | float | int) -> Tensor:
    class ReLU(BaseOperation):
        def _forward(self):
            return self._get_arg(0) * np.where(self._get_arg(0) > 0, 1, 0)
    
        def _backward(self, i):
            return np.where(self._get_arg(i) > 0, 1, 0)
        
        def info(self, verbose=False):
            return f"ReLU ({self._join_args(verbose=verbose)})"
    
    op = ReLU()
    return op.forward(x)

def sum_along_axis_0(x: Tensor) -> Tensor:
    acc = Tensor(np.zeros_like(x[0]))
    for val in x:
        acc += val
    return acc

def tanh(x: Tensor | float | int) -> Tensor:
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def softmax(x: Tensor) -> Tensor:
    denom = sum_along_axis_0(exp(x))
    return exp(x) / denom

def mean_squared_error(x: Tensor, y: Tensor) -> Tensor:
    return (1 / x.shape[0]) * sum_along_axis_0((x - y) ** 2)

def cross_entropy(x: Tensor, y: Tensor) -> Tensor:
    op = NaturalLogarithm()
    return -sum_along_axis_0(y * op.forward(x))
