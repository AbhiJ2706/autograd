from __future__ import annotations

import numpy as np

import mytorch.operation as operation


class Tensor:
    class Backlink:
        def __init__(self, indexer: int | slice, referee: Tensor):
            self.indexer = indexer
            self.referee = referee

        def send(self, grad: np.ndarray):
            referee_grad = np.zeros_like(self.referee.val, dtype=float)
            referee_grad[self.indexer] = grad.reshape(referee_grad[self.indexer].shape)
            self.referee.backward(referee_grad)
        
        def __str__(self):
            return f"Backlink ({self.indexer} referee=@{hex(id(self.referee))})"

    def __init__(
        self, 
        val: np.ndarray | float | int | list, 
        creator: operation.BaseOperation = None, 
        backlink: Tensor.Backlink = None
    ):
        if isinstance(val, np.ndarray):
            self.val: np.ndarray = val.astype(np.float32)
        elif isinstance(val, list):
            self.val: np.ndarray = np.array(val, dtype=np.float32)
        else:
            self.val: np.ndarray = np.array([val], dtype=np.float32)

        if len(self.val.shape) == 0:
            self.val = self.val.reshape((1,))
        self.creator: operation.BaseOperation = creator
        self.grad: np.ndarray | None = None
        self.backlink: Tensor.Backlink = backlink
    
    def zero_grad(self) -> None:
        if self.creator:
            self.creator.zero_grad()
        self.__clean()

    def backward(self, current_gradient: np.ndarray | None = None) -> None:
        if self.grad is None:
            if current_gradient is None:
                current_gradient = np.ones_like(self.val)
            axis_list = self.__resolve_shape(current_gradient)
            if len(axis_list):
                self.grad = np.sum(current_gradient, axis=tuple(axis_list)).reshape(self.val.shape)
            else:
                self.grad = current_gradient
        else:
            axis_list = self.__resolve_shape(current_gradient)
            if len(axis_list):
                self.grad += np.sum(current_gradient, axis=tuple(axis_list)).reshape(self.val.shape)
            else:
                self.grad += current_gradient
        if self.backlink:
            self.backlink.send(self.grad)
        if self.creator:
            self.creator.backward(current_gradient)
    
    def to_numpy(self) -> np.ndarray:
        return self.val
    
    def info(self, verbose=False):
        if self.creator is not None:
            return self.creator.info(verbose=verbose)
        else:
            if verbose:
                return f"Tensor @{hex(id(self))} ({str(self.val)} grad={self.grad} backlink={self.backlink})"
            return str(self)

    @property
    def shape(self):
        if isinstance(self.val, np.ndarray):
            return tuple(self.val.shape)
        return (1,)

    def __resolve_shape(self, grad: np.ndarray):
        self_shape = self.val.shape
        grad_shape = grad.shape
        padded_self_shape = (1,) * (len(grad_shape) - len(self_shape)) + self_shape
        axis_list = []
        for i, (s_dim, g_dim) in enumerate(zip(padded_self_shape, grad_shape)):
            if s_dim == 1 and g_dim != 1:
                axis_list.append(i)
        return axis_list

    def direct_update(self, new: Tensor):
        self.__clean()
        self.val = new.val

    def __clean(self):
        self.grad = None
        self.backlink = None
        self.creator = None
    
    def __str__(self):
        return f"Tensor @{hex(id(self))} ({str(self.val)})"
    
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
    
    def __matmul__(self, other: Tensor) -> Tensor:
        if len(self.shape) == 1 and len(other.shape) == 1:
            op = operation.DotProduct()
        else:
            op = operation.MatrixMultiplication()
        return op.forward(self, other)

    def __abs__(self) -> Tensor:
        op = operation.AbsoluteValue()
        return op.forward(self)

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
    
    def __getitem__(self, i: int) -> Tensor:
        return Tensor(self.val[i], backlink=Tensor.Backlink(i, self))
