import math
import numpy as np
from mytorch.tensor import Tensor


class Dense:
    def __init__(self, n_in: int, n_out: int):
        weight_initialization = lambda n, size: np.random.uniform(-1 / math.sqrt(n), 1 / math.sqrt(n), size=size)
        self.W = Tensor(weight_initialization(n_out, (n_out, n_in)))
        self.b = Tensor(np.zeros(shape=(n_out,)))
    
    def forward(self, X: Tensor) -> Tensor:
        return self.W @ X + self.b
