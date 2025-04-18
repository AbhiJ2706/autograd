from __future__ import annotations

import numpy as np

import autograd.operation as operation


class Tensor:
    def __init__(self, val: np.ndarray | float | int | list, creator: 'operation.BaseOperation' = None):
        if isinstance(val, np.ndarray):
            self.val: np.ndarray = val
        elif isinstance(val, list):
            self.val: np.ndarray = np.array(val)
        else:
            self.val: np.ndarray = np.array([val])
        self.creator: operation.BaseOperation = creator
        self.grad: np.ndarray | None = None
    
    def zero_grad(self) -> None:
        self.grad = None
        if self.creator:
            self.creator.zero_grad()

    def backward(self, current_gradient: np.ndarray | None = None) -> None:
        if self.grad is None:
            if current_gradient is None:
                current_gradient = np.ones_like(self.val)
            self.grad = current_gradient
        else:
            self.grad += current_gradient
        if self.creator:
            self.creator.backward(current_gradient)
    
    def to_numpy(self) -> np.ndarray:
        return self.val
    
    def info(self, verbose=False):
        if self.creator is not None:
            return self.creator.info(verbose=verbose)
        else:
            if verbose:
                return f"Tensor({str(self.val)} grad={self.grad})"
            return str(self)

    @property
    def shape(self):
        if isinstance(self.val, np.ndarray):
            return tuple(self.val.shape)
        return (1,)
    
    def __str__(self):
        return f"Tensor({str(self.val)})"
    
    def __add__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Add()
        return op.forward(self, other)

    def __iadd__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Add()
        return op.forward(self, other)
    
    def __radd__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Add()
        return op.forward(other, self)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Subtract()
        return op.forward(self, other)

    def __isub__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Subtract()
        return op.forward(self, other)
    
    def __rsub__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Subtract()
        return op.forward(other, self)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Multiply()
        return op.forward(self, other)

    def __imul__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Multiply()
        return op.forward(self, other)
    
    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Multiply()
        return op.forward(other, self)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Divide()
        return op.forward(self, other)
    
    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Divide()
        return op.forward(other, self)

    def __idiv__(self, other: Tensor | float | int) -> Tensor:
        op = operation.Divide()
        return op.forward(self, other)

    def __pow__(self, other: float | int) -> Tensor:
        op = operation.Exponent()
        return op.forward(self, other)

    def __ipow__(self, other: float | int) -> Tensor:
        op = operation.Exponent()
        return op.forward(self, other)
    
    def __rpow__(self, other: float | int) -> Tensor:
        op = operation.Exponent()
        return op.forward(other, self)
    
    def __matmul__(self, other: Tensor):
        op = operation.MatrixMultiplication()
        return op.forward(self, other)

    def __neg__(self) -> Tensor:
        op = operation.Multiply()
        return op.forward(self, -1)

    def __lt__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self.val < other.val
        return self.val < other

    def __gt__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self.val > other.val
        return self.val > other

    def __le__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self.val <= other.val
        return self.val <= other

    def __ge__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self.val >= other.val
        return self.val >= other

    def __ne__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self.val != other.val
        return self.val != other

    def __eq__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self.val == other.val
        return self.val == other
