from __future__ import annotations

import abc

import numpy as np

import autograd.tensor as tensor


class BaseOperation(abc.ABC):
    def __init__(self):
        self.args: list[tensor.Tensor | float | int] = []
        self.arg_gradients: list[np.ndarray | float | int] = []

    def backward(self, current_gradient: np.ndarray):
        for i, arg in enumerate(self.args):
            gradient = current_gradient * self._backward(i)
            if isinstance(arg, tensor.Tensor):
                arg.backward(gradient)
            self.arg_gradients.append(gradient)
    
    def forward(self, *args):
        self.args = args
        result = tensor.Tensor(self._forward())
        result.creator = self
        return result
    
    def zero_grad(self):
        for arg in self.args:
            arg.zero_grad()
    
    @abc.abstractmethod
    def info(self):
        raise NotImplementedError("Not implemented")
    
    def _get_arg(self, i):
        return BaseOperation._get_or_else(self.args[i])
    
    @abc.abstractmethod
    def _backward(self, gradient: np.ndarray, index: int) -> np.ndarray:
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def _forward(self) -> tensor.Tensor:
        raise NotImplementedError("Not implemented")
    
    @staticmethod
    def _get_or_else(arg: tensor.Tensor | float | int) -> tensor.Tensor | float | int:
        if isinstance(arg, tensor.Tensor):
            return arg.val
        else:
            return arg
        
    def _join_args(self, verbose=False):
        joined = []
        if not len(self.arg_gradients):
            self.arg_gradients = [None for _ in self.args]
        for g, a in zip(self.arg_gradients, self.args):
            if isinstance(a, tensor.Tensor):
                joined.append(a.info(verbose=verbose))
            else:
                if verbose:
                    joined.append(str(a) + f" grad={g}")
                else:
                    joined.append(str(a))
        return ' '.join(joined)
    
    def __str__(self):
        return self.info()
        
    
class Add(BaseOperation):
    def _forward(self):
        return self._get_arg(0) + self._get_arg(1)
    
    def _backward(self, i):
        return np.ones_like(self._get_arg(i))
    
    def info(self, verbose=False):
        return f"Add ({self._join_args(verbose=verbose)})"
    

class Subtract(BaseOperation):
    def _forward(self):
        return self._get_arg(0) - self._get_arg(1)
    
    def _backward(self, i):
        return np.ones_like(self._get_arg(i)) * (-1 if i else 1)
    
    def info(self, verbose=False):
        return f"Subtract ({self._join_args(verbose=verbose)})"
    

class Multiply(BaseOperation):
    def _forward(self):
        return self._get_arg(0) * self._get_arg(1)
    
    def _backward(self, i):
        return self._get_arg(1 - i)
    
    def info(self, verbose=False):
        return f"Multiply ({self._join_args(verbose=verbose)})"
    

class Divide(BaseOperation):
    def _forward(self):
        return self._get_arg(0) / self._get_arg(1)
    
    def _backward(self, i):
        if i == 0:
            return 1 / self._get_arg(1)
        return -1 * self._get_arg(0) * (self._get_arg(1) ** -2)
    
    def info(self, verbose=False):
        return f"Divide ({self._join_args(verbose=verbose)})"


class Exponent(BaseOperation):
    def _forward(self):
        return self._get_arg(0) ** self._get_arg(1)
    
    def _backward(self, i):
        if i == 0:
            return self._get_arg(1) * self._get_arg(0) ** (self._get_arg(1) - 1)
        return self._get_arg(0) ** self._get_arg(1) * np.log(self._get_arg(0))
    
    def info(self, verbose=False):
        return f"Exponent ({self._join_args(verbose=verbose)})"
    

class MatrixMultiplication(BaseOperation):
    def _forward(self):
        return self.args[0].val @ self.args[1].val
    
    def backward(self, current_gradient: np.ndarray):
        self.args[0].backward(current_gradient @ self.args[1].val.T)
        self.args[1].backward(self.args[0].val.T @ current_gradient)
    
    def _backward(self):
        pass
    
    def info(self, verbose=False):
        return f"Exponent ({self._join_args(verbose=verbose)})"
    

class NaturalLogarithm(BaseOperation):
    def _forward(self):
        return np.log(self._get_arg(0))
    
    def _backward(self, i):
        return 1 / self._get_arg(0)
    
    def info(self, verbose=False):
        return f"Ln ({self._join_args(verbose=verbose)})"
    