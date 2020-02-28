import unittest

import numpy as np

from tinynet.loss import CrossEntropy


class TestTensorCrossEntropy(unittest.TestCase):

    def test_simple_cross_entropy(self):
        predicted = np.array([[1, 1, 1], [0, 0, 0]])
        actual = np.array([1, 0])
        criterion = CrossEntropy()
        loss = criterion(predicted, actual)
        print('[tinynet] loss', loss)

        np.testing.assert_almost_equal(loss, -2 * np.log(1 / 3))

        grad = criterion.backward(predicted, actual)
        np.testing.assert_array_almost_equal(
            grad,
            np.array([[1 / 3, -2 / 3, 1 / 3], [-2 / 3, 1 / 3, 1 / 3]])
        )

        import torch
        from torch import nn
        torch_criterion = nn.CrossEntropyLoss(reduction="sum")
        torch_predicted = torch.FloatTensor(predicted)
        torch_predicted.requires_grad = True

        torch_actual = torch.LongTensor(actual)
        torch_loss = torch_criterion.forward(torch_predicted, torch_actual)

        # assert np.testing.assert_almost_equal(loss, torch_loss.item())
        torch_loss.backward()
        print("torch loss:", torch_loss.item())
        print("torch grad:", torch_predicted.grad)

        np.testing.assert_allclose(grad, torch_predicted.grad.data)

    def test_cross_entropy(self):

        np.random.seed(1)
        predicted = np.random.rand(2, 3)
        actual = np.array([1, 0])
        criterion = CrossEntropy()
        loss = criterion(predicted, actual)
        print('[tinynet] loss:', loss)

        grad = criterion.backward(predicted, actual)
        print("[tinynet] grad:", grad)

        import torch
        from torch import nn
        torch_criterion = nn.CrossEntropyLoss(reduction="sum")
        torch_predicted = torch.FloatTensor(predicted)
        torch_predicted.requires_grad = True

        torch_actual = torch.LongTensor(actual)
        torch_loss = torch_criterion.forward(torch_predicted, torch_actual)

        np.testing.assert_almost_equal(loss, torch_loss.item())
        torch_loss.backward()
        print("torch loss:", torch_loss.item())
        print("torch grad:", torch_predicted.grad)

        np.testing.assert_allclose(grad, torch_predicted.grad.data)

    def test_brutal_cross_entropy(self):

        np.random.seed(1)

        for _ in range(10):
            self.test_cross_entropy()
