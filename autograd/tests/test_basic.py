import numpy as np
import pytest

from autograd.tensor import Tensor


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


def test_exponent():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    (x ** 2).backward()
    assert np.all(x ** 2 == [
        [1, 4, 9], [16, 25, 36]
    ])
    assert np.all(x.grad ==[
        [2, 4, 6], [8, 10, 12]
    ])
    
