import numpy as np


class DefaultOptimizer:
    def __init__(self, *args, lr=0.01):
        self.args = list(args)
        self.lr = lr
        self.arg_grads = [np.zeros_like(a.val) for a in self.args]
        self.update_counter = 0
    
    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        pass

    def step(self):
        self.update_counter += 1
        for i, arg in enumerate(self.args):
            self.arg_grads[i] += self.lr * arg.grad
    
    def zero_grad(self):
        if not self.update_counter: return self.args
        for i, update in enumerate(self.arg_grads):
            self.args[i] -= update / self.update_counter
            self.args[i].clean()
        self.arg_grads = [np.zeros_like(a.val) for a in self.args]
        self.update_counter = 0
        return self.args
