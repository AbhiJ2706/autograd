import numpy as np
from autograd.tensor import Tensor
import autograd.functional as F


if __name__ == "__main__":
    x = Tensor(3)
    y = Tensor(4)
    print(x + y)
    print((x + y).info())
    (x + y).backward()
    print(x.grad, y.grad)
    print(x.shape)

    print("-------------")

    x = Tensor(3)
    y = Tensor(4)
    print(x * y)
    print((x * y).info())
    (x * y).backward()
    print(x.grad, y.grad)

    print("-------------")

    x = Tensor(3)
    y = Tensor(4)
    z = Tensor(5)
    print((x + y) * z)
    print(((x + y) * z).info())
    ((x + y) * z).backward()
    print(x.grad, y.grad, z.grad)

    print("-------------")

    x = Tensor(3)
    print((3 * x))
    print((3 * x).info())
    (3 * x).backward()
    print(x.grad)

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = Tensor([[7, 8, 9], [10, 11, 12]])
    print(x + y)
    print((x + y).info())
    (x + y).backward()
    print(x.grad, y.grad)

    print("-------------")

    x.zero_grad()
    y.zero_grad()
    print(x * y)
    print((x * y).info())
    (x * y).backward()
    print(x.grad, y.grad)

    print("-------------")

    x.zero_grad()
    y.zero_grad()
    print(2 * x * y)
    print((2 * x * y).info())
    (2 * x * y).backward()
    print(x.grad, y.grad)

    print("-------------")

    x.zero_grad()
    y.zero_grad()
    print(x ** 2)
    print((x ** 2).info())
    (x ** 2).backward()
    print(x.grad)

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = Tensor(np.array([[7, 8, 9], [10, 11, 12]]).T)
    print(x @ y)
    print((x @ y).info())
    (x @ y).backward()
    print(x.grad, y.grad)

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    print(1 - x)
    print((1 - x).info())
    (1 - x).backward()
    print(x.grad)

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    print(-x)
    print((-x).info())
    (-x).backward()
    print(x.grad)

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    print(2 ** x)
    print((2 ** x).info())
    (2 ** x).backward()
    print(x.grad)

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    print(F.exp(x))
    print(F.exp(x).info())
    F.exp(x).backward()
    print(x.grad)

    print("-------------")

    x = Tensor([[-3, -2, -1], [1, 2, 3]])
    print(F.sigmoid(x))
    y = F.sigmoid(x)
    y.backward()
    print(y.info())
    print(x.grad)

    print("-------------")

    x = Tensor([[-3, -2, -1], [1, 2, 3]])
    print(F.relu(x))
    y = F.relu(x)
    y.backward()
    print(y.info())
    print(x.grad)
    print(x.shape)
    print(F.relu(4))

    print("-------------")

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = Tensor([[7, 8, 9], [10, 11, 12]])
    print(F.mean_squared_error(x, y))
    z = F.mean_squared_error(x, y)
    z.backward()
    print(z.info(verbose=True))
    print(x.grad, y.grad)

    print("-------------")

    x = Tensor([1, 2, 3, 4, 5])
    print(F.softmax(x))
    z = F.softmax(x)
    z.backward()
    print(z.info())
    print(x.grad)
