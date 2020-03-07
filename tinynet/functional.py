import numpy as np
from scipy.special import erf
from .tensor import Tensor


def tanh(t: Tensor) -> Tensor:
    data = np.tanh(t.data)
    return data


def phi(t: Tensor) -> Tensor:
    from scipy.stats import norm
    z = norm.cdf(t)
    return z


def d_erf(t: Tensor) -> Tensor:
    z = 2/np.sqrt(np.pi) * np.exp(- t ** 2)
    return z


def d_phi(t: Tensor) -> Tensor:
    from scipy.stats import norm
    z = norm.pdf(t)
    return z


def gelu(t: Tensor) -> Tensor:
    return t * phi(t)


def d_gelu(t: Tensor) -> Tensor:
    z = phi(t) + t * d_phi(t)
    return z


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
