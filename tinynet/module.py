from typing import Dict, Iterator, List
import inspect
from collections import OrderedDict

import numpy as np
from .tensor import Tensor
from .layers import Layer
from .parameter import Parameter


class Module:

    def __init__(self) -> None:
        self.training = True
        self.layers = OrderedDict()

    # def parameters(self) -> Iterator[Parameter]:
    #
    #     for name, value in inspect.getmembers(self):
    #         if isinstance(value, Parameter):
    #             yield value
    #         elif isinstance(value, Module):
    #             yield from value.parameters()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def zero_grad(self) -> None:
        #
        # for name, param in self.parameters():
        #     if param.backward:
        #         param.backward = np.zeros_like(param.backward)
        #
        for name, value in inspect.getmembers(self):
            if isinstance(value, Module):
                value.zero_grad()

    def parameters(self):

        import inspect
        for name, value in inspect.getmembers(self):
            # print(f"{name}: {value}", type(value))
            if isinstance(value, Layer):
                print(f"{name}: {value}", type(value), value.grads, value.params)

                yield (value.params, value.grads)
            elif isinstance(value, Module):
                yield from value.parameters()


class Sequential(Module):

    def __init__(self, layers: List[Layer]) -> None:
        super().__init__()

        self.layers: List[Layer] = layers if layers else []

    def add(self, layer: Layer) -> 'Sequential':
        self.layers.append(layer)
        return self

    def __call__(self, inputs) -> Tensor:

        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
