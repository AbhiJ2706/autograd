import numpy as np
from mytorch.tensor import Tensor
import mytorch.functional as F


def test_mean_squared_error():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = Tensor([[7, 8, 9], [10, 11, 12]])
    assert np.allclose(F.mean_squared_error(x, y).to_numpy(), [[36, 36, 36], [36, 36, 36]])
    z = F.mean_squared_error(x, y)
    z.backward()
    