from typing import Tuple
from collections import defaultdict
import numpy as np

from .optimizer import Optimizer
from tinynet import Sequential
from tinynet.tensor import Tensor


class AdamMax(Optimizer):

    def __init__(self, model: Sequential, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0,
                 eps: float = 1e-8) -> None:

        self.model = model
        self.defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        self.state = defaultdict(dict)

    def step(self, grad: Tensor):

        lr = self.defaults["lr"]
        weight_decay = self.defaults["weight_decay"]
        eps = self.defaults["eps"]

        for layer in self.model.layers.layers:

            for name, param in layer.params.items():
                grad = layer.grads[name]

                state = self.state[id(param)]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(grad)
                    state["exp_inf"] = np.zeros_like(grad)

                exp_avg, exp_inf = state["exp_avg"], state["exp_inf"]
                beta1, beta2 = self.defaults["betas"]
                state["step"] += 1

                if weight_decay:
                    grad += weight_decay * param

                bias_correction1 = 1 - beta1 ** state["step"]
                exp_avg = state["exp_avg"] = exp_avg * beta1 + (1 - beta1) * grad
                exp_inf = state["exp_inf"] = np.fmax(exp_inf * beta2, np.abs(grad) + eps)

                exp_avg = exp_avg / bias_correction1

                param -= lr * exp_avg / (exp_inf + eps)
