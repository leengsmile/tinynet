from collections import defaultdict
import numpy as np

from .optimizer import Optimizer
from tinynet import Sequential
from tinynet.tensor import Tensor


class RMSprop(Optimizer):

    def __init__(self, model: Sequential, lr: float = 0.01, alpha: float = 0.99, momentum: float = 0,
                 eps: float = 1e-8, weight_decay: float = 0, centered: bool = False) -> None:

        self.model = model
        self.defaults = dict(lr=lr, alpha=alpha, momentum=momentum,
                             weight_decay=weight_decay, eps=eps, centered=centered)
        self.state = defaultdict(dict)

    def step(self, grad: Tensor):

        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]
        centered = self.defaults["centered"]
        eps = self.defaults["eps"]

        for layer in self.model.layers.layers:

            for name, param in layer.params.items():
                grad = layer.grads[name]
                state = self.state[id(param)]
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = np.zeros_like(param)
                    if momentum:
                        state["momentum_buf"] = np.zeros_like(param)
                    if centered:
                        state["grad_avg"] = np.zeros_like(param)

                state["step"] += 1

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                square_avg = state["square_avg"]
                square_avg = state["square_avg"] = alpha * square_avg + (1 - alpha) * grad ** 2

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg = state["grad_avg"] = alpha * grad_avg + (1 - alpha) * grad
                    avg = np.sqrt(square_avg - grad_avg ** 2) + eps
                else:
                    avg = np.sqrt(square_avg) + eps

                if momentum != 0:
                    buf = state["momentum_buf"]
                    buf = state["momentum_buf"] = momentum * buf + grad / avg
                else:
                    buf = grad / avg

                param -= lr * buf


