from mytorch.tensor import Tensor
from mytorch.operation import *


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

def conv1d(x: Tensor, y: Tensor, padding: int = 0, stride: int = 1) -> Tensor:
    class Conv1d(BaseOperation):
        def _forward(self):
            f = self._get_arg(0)
            k = self._get_arg(1)
            return np.convolve(f, np.flip(k), mode='valid')
        
        def backward(self, current_gradient: np.ndarray):
            k = self._get_arg(1)
            f = self._get_arg(0)

            self.args[0].backward(np.convolve(current_gradient, k, mode='full'))
            self.args[1].backward(np.convolve(f, current_gradient, mode='valid'))
    
        def _backward(self, i):
            pass
        
        def info(self, verbose=False):
            return f"Conv1d ({self._join_args(verbose=verbose)})"
    
    op = Conv1d()
    return op.forward(x, y)

