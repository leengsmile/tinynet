from typing import Dict, List
import numpy as np
from .tensor import Tensor
from .functional import tanh


class Layer:

    def __init__(self):
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.training = True

    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self):
        for name, param in self.params.items():
            grad = self.grads[name]
            yield param, grad

    def zero_grad(self):
        for name, param in self.params.items():
            self.grads[name] = np.zeros_like(param)


class Tanh(Layer):

    __slots__ = ["inputs"]

    def __call__(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return tanh(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        y = tanh(self.inputs)
        return grad * (1 - y * y)


class Linear(Layer):

    def __init__(self, input_shape: int, output_shape: int, bias: bool = True) -> None:
        super().__init__()
        a = np.sqrt(6) / np.sqrt(input_shape + output_shape)
        self.params["w"] = np.random.uniform(-a, a, (input_shape, output_shape))
        if bias:
            a = np.sqrt(6) / np.sqrt(output_shape)
            self.params["b"] = np.random.uniform(-a, a, (output_shape))
        else:
            self.params["b"] = None

    def __call__(self, inputs: Tensor) -> Tensor:
        self.inputs: Tensor = inputs
        z = inputs @ self.params["w"] + self.params["b"]
        return z

    def backward(self, grad: Tensor) -> Tensor:
        """
        z = x @ w + b
        given dl/dz
        dl/dw = x^T @ backward
        dl/dx = backward @ w^T
        dl/db = ??
        """

        # self.weights.backward += self.inputs.data.T @ backward.data
        # self.bias.backward += backward.data.sum(axis=0)
        self.grads["w"] += self.inputs.T @ grad
        self.grads["b"] += grad.sum(axis=0)
        return grad @ self.params["w"].T
        # return backward @ self.params["w"].T


class Relu(Layer):

    def __call__(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        outputs = inputs.copy()
        outputs[outputs < 0] = 0
        return outputs

    def backward(self, grad: Tensor) -> Tensor:

        grad = grad.copy()
        grad[self.inputs < 0] = 0
        return grad


class Dropout(Layer):

    def __init__(self, dropout: float = 0.):
        super().__init__()
        assert 0 <= dropout <= 1, "dropout should be in [0, 1]."
        self.dropout = dropout

    def __call__(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        mask = np.random.rand(inputs.shape)
        mask = mask < self.dropout
        self.mask = mask
        output = inputs.copy()
        output[mask < self.dropout] = 0
        if self.training:
            output /= (1 - self.dropout)
        return output

    def backward(self, grad: Tensor) -> Tensor:

        mask = self.mask
        grad = grad.copy()
        grad[mask] = 0
        return grad

