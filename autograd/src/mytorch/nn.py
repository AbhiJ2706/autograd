import abc
import math
import numpy as np
from mytorch.tensor import Tensor
import mytorch.functional as F


class Module(abc.ABC):
    def __call__(self, *args):
        return self.forward(*args)
    
    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        raise NotImplementedError("not implemented")
    
    @abc.abstractmethod
    def parameters(self) -> list[Tensor]:
        raise NotImplementedError("not implemented")


class Dense(Module):
    def __init__(self, n_in: int, n_out: int):
        weight_initialization = lambda n, size: np.random.uniform(-1 / math.sqrt(n), 1 / math.sqrt(n), size=size)
        self.W = Tensor(weight_initialization(n_out, (n_out, n_in)))
        self.b = Tensor(np.zeros(shape=(n_out,)))
    
    def forward(self, X: Tensor) -> Tensor:
        return self.W @ X + self.b
    
    def parameters(self):
        return [self.W, self.b]
    

class Sigmoid(Module):
    def forward(self, X: Tensor) -> Tensor:
        return F.sigmoid(X)
    
    def parameters(self):
        return []


class Softmax(Module):
    def forward(self, X: Tensor) -> Tensor:
        return F.softmax(X)
    
    def parameters(self):
        return []
    