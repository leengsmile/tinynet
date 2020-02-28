import numpy as np
from .tensor import Tensor
from .functional import softmax, log_softmax, onehot


class Loss:

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        error = predicted - actual
        return error * error

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class CrossEntropy(Loss):

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """
        predicted: (batch, k)
        actual: (batch, )
        """

        log_probs = log_softmax(predicted, axis=1)
        indices = range(len(actual))
        log_loss = log_probs[indices, actual]
        total_loss = - np.sum(log_loss)
        return total_loss

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:

        probs = softmax(predicted, axis=1)
        _, k = predicted.shape
        actual = onehot(actual, k)
        return probs - actual

