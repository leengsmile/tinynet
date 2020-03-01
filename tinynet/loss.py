import numpy as np
from .tensor import Tensor
from .functional import softmax, log_softmax, onehot


class Loss:

    def __init__(self, reduction: bool = False):
        self.reduction = reduction

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):

    def __init__(self, reduction: bool = False):
        super().__init__(reduction)

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        error = predicted - actual
        return error * error

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class CrossEntropy(Loss):

    def __init__(self, reduction: bool = False):
        super().__init__(reduction)

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """
        predicted: (batch, k)
        actual: (batch, )
        """

        log_probs = log_softmax(predicted, axis=1)
        indices = range(len(actual))
        log_loss = log_probs[indices, actual]
        total_loss = - np.sum(log_loss)
        if self.reduction:
            total_loss /= len(actual)
        return total_loss

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:

        probs = softmax(predicted, axis=1)
        _, k = predicted.shape
        actual = onehot(actual, k)
        grad = probs - actual
        if self.reduction:
            grad /= len(actual)
        return grad
