from collections import defaultdict
import numpy as np

from .optimizer import Optimizer
from tinynet import Sequential
from tinynet.tensor import Tensor


class Adagrad(Optimizer):

    def __init__(self, model: Sequential, lr: float = 0.01, weight_decay: float = 0,
                 initial_accumulator_value: float = 0, eps: float = 1e-10) -> None:

        self.model = model
        self.defaults = dict(lr=lr, weight_decay=weight_decay,
                             initial_accumulator_value=initial_accumulator_value, eps=eps)
        self.state = defaultdict(dict)

    def step(self, grad: Tensor):

        lr = self.defaults["lr"]
        weight_decay = self.defaults["weight_decay"]
        initial_accumulator_value = self.defaults["initial_accumulator_value"]
        eps = self.defaults["eps"]

        for layer in self.model.layers.layers:

            for name, param in layer.params.items():
                grad = layer.grads[name]
                state = self.state[id(param)]
                if "sum" not in state:
                    state["sum"] = np.full_like(param, initial_accumulator_value)

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                state["sum"] += grad ** 2
                std = np.sqrt(state["sum"])
                param -= lr * grad / (std + eps)


