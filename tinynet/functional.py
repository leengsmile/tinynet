import numpy as np
from .tensor import Tensor


def tanh(t: Tensor) -> Tensor:
    data = np.tanh(t.data)
    return data


def softmax(t: Tensor, axis=None) -> Tensor:
    return np.exp(t - log_sumexp(t, axis=axis, keepdims=True))


def log_softmax(t: Tensor, axis=None) -> Tensor:
    return t - log_sumexp(t, axis=axis, keepdims=True)


def log_sumexp(t: Tensor, axis: int=None, keepdims: bool=False) -> Tensor:
    t_max = t.max(axis=axis, keepdims=keepdims)
    z = np.exp(t - t_max)
    z = np.sum(z, axis=axis, keepdims=keepdims)
    z = np.log(z)
    z += t_max
    return z


def onehot(t: Tensor, k: int) -> Tensor:
    n = len(t)
    encode = np.zeros((n, k))
    encode[range(n), t] = 1
    return encode
