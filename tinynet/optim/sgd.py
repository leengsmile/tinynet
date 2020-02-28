from collections import defaultdict
import numpy as np

from .optimizer import Optimizer
from tinynet import Sequential
from tinynet.tensor import Tensor


class SGD(Optimizer):

    def __init__(self, model: Sequential, lr: float = 0.01, momentum: float = 0, nesterov: bool = True) -> None:

        self.model = model
        self.defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        self.state = defaultdict(dict)

    def step(self, grad: Tensor):

        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        nesterov = self.defaults["nesterov"]

        for layer in self.model.layers.layers:

            for name, param in layer.params.items():
                grad = layer.grads[name]

                if momentum != 0:
                    param_state = self.state[id(param)]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = np.zeros_like(grad)
                        buf = param_state["momentum_buffer"] = momentum * buf + grad
                    else:
                        buf = param_state["momentum_buffer"]
                        buf = param_state["momentum_buffer"] = momentum * buf + grad
                    if nesterov:
                        grad = momentum * buf + grad
                    else:
                        grad = buf

                param -= lr * grad


