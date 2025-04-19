import numpy as np
import pytest

from mytorch.functional import exp, sum_along_axis_0
from mytorch.tensor import Tensor


def test_add():
    x = Tensor(3)
    y = Tensor(4)
    assert x + y == Tensor([7])
    (x + y).backward()
    assert x.grad == np.array([1])
    assert y.grad == np.array([1])
    assert x.shape == (1,)

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = Tensor([[7, 8, 9], [10, 11, 12]])
    assert np.all(x + y == Tensor([
        [8, 10, 12], [14, 16, 18]
    ]))
    (x + y).backward()
    assert np.all(x.grad == np.ones_like(x.grad))
    assert np.all(y.grad == np.ones_like(y.grad))


def test_subtract():
    x = Tensor(3)
    y = Tensor(4)
    assert x - y == Tensor([-1])
    (x - y).backward()
    assert x.grad == np.array([1])
    assert y.grad == np.array([-1])
    assert x.shape == (1,)

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    assert np.all(1 - x == [[0, -1, -2], [-3, -4, -5]])
    (1 - x).backward()
    assert np.all(x.grad == [[-1, -1, -1], [-1, -1, -1]])


def test_multiply():
    x = Tensor(3)
    y = Tensor(4)
    assert x * y == Tensor([12])
    (x * y).backward()
    assert x.grad == np.array([4])
    assert y.grad == np.array([3])
    assert x.shape == (1,)


def test_constant_multiply():
    x = Tensor(3)
    assert 3 * x == Tensor([9])
    (3 * x).backward()
    assert x.grad == np.array([3])

    x.zero_grad()
    assert -x == Tensor([-3])
    (-x).backward()
    assert x.grad == np.array([-1])


def test_exponent():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    (x ** 2).backward()
    assert np.all(x ** 2 == [
        [1, 4, 9], [16, 25, 36]
    ])
    assert np.all(x.grad ==[
        [2, 4, 6], [8, 10, 12]
    ])

    x.zero_grad()
    assert np.all(2 ** x == [
        [2, 4, 8], [16, 32, 64]
    ])
    (2 ** x).backward()
    assert np.all(x.grad == 2 ** np.array([
        [1, 2, 3], [4, 5, 6]
    ]) * np.log(2))


def test_slice():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = x[1:2]
    (3 * y).backward()
    assert np.all(x.grad == [[0, 0, 0], [3, 3, 3]])


def test_slice_one_dimension():
    x = Tensor([1, 2, 3, 4, 5])
    y = x[1:2]
    exp(y).backward()
    assert np.allclose(x.grad, [0, np.exp(2), 0, 0, 0])


def test_slice_multiple_dimensions():
    x = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y = x[0, :, 0]
    assert np.all(y == [1, 4])
    exp(y).backward()
    assert np.allclose(x.grad, [
        [[ 2.71828183, 0, 0],
         [54.59815003, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0]]
    ])


def test_broadcast():
    x = Tensor([1, 2, 3, 4, 5])
    y = Tensor(4)
    (x / y ** 2).backward()
    assert np.allclose(x.grad, [1 / 16, 1 / 16, 1 / 16, 1 / 16, 1 / 16])
    assert np.allclose(y.grad, sum([-z / 32 for z in [1, 2, 3, 4, 5]]))


def test_broadcast_multiple_dimensions():
    x = Tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
    y = Tensor([[2], [4]])
    (x / y ** 2).backward()
    assert np.allclose(x.grad, [[0.25, 0.25, 0.25, 0.25], [1 / 16, 1 / 16, 1 / 16, 1 / 16]])
    assert np.allclose(y.grad, [[-2.5], [-0.5625]])


def test_broadcast_multiple_operations():
    x = Tensor([1, 2, 3, 4])
    (exp(x) / sum_along_axis_0(exp(x))).backward()
    assert np.allclose(x.grad, [0, 0, 0, 0])


def test_matrix_multiplication():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = Tensor(np.array([[7, 8, 9], [10, 11, 12]]).T)
    assert np.all(x @ y == [[50, 68], [122, 167]])
    (x @ y).backward()
    assert np.allclose(x.grad, [[17, 19, 21], [17, 19, 21]])
    assert np.allclose(y.grad, [[5, 5], [7, 7], [9, 9]])
