from collections import defaultdict
import numpy as np

from .optimizer import Optimizer
from tinynet import Sequential
from tinynet.tensor import Tensor


class Adadelta(Optimizer):

    def __init__(self, model: Sequential, lr: float = 1, rho: float = 0.9,
                 weight_decay: float = 0, eps: float = 1e-6) -> None:

        self.model = model
        self.defaults = dict(lr=lr, rho=rho, weight_decay=weight_decay, eps=eps)
        self.state = defaultdict(dict)

    def step(self, grad: Tensor):

        lr = self.defaults["lr"]
        weight_decay = self.defaults["weight_decay"]
        rho = self.defaults["rho"]
        eps = self.defaults["eps"]

        for layer in self.model.layers.layers:

            for name, param in layer.params.items():
                grad = layer.grads[name]
                state = self.state[id(param)]
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = np.zeros_like(param)
                    state["acc_delta"] = np.zeros_like(param)

                state["step"] += 1

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                square_avg, acc_delta = state["square_avg"], state["acc_delta"]
                square_avg = rho * square_avg + (1 - rho) * grad ** 2
                delta = np.sqrt(acc_delta + eps) / (np.sqrt(square_avg + eps)) * grad  # pytorch update rule

                param -= lr * delta

                state["square_avg"] = square_avg
                state["acc_delta"] = rho * acc_delta + (1 - rho) * delta ** 2

