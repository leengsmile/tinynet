import numpy as np
import torch
from torch import nn
from tinynet.tensor import Tensor
from tinynet.layers import GELU
import unittest


class TestTensorGELULayer(unittest.TestCase):

    def test_tensor_gelu_simple(self):

        inputs = np.array([[-1, 0, 1]])
        gelu = GELU()
        out = gelu(inputs)
        print("output:", out)

        torch_inputs = torch.tensor(inputs, dtype=torch.float)
        torch_inputs.requires_grad = True
        torch_gelu = nn.GELU()
        torch_out = torch_gelu.forward(torch_inputs)
        print("torch output:", torch_out.detach().numpy())

        np.testing.assert_allclose(out, torch_out.detach().numpy(), 1e-7, 1e-6)

        z_grad = np.ones_like(inputs)
        grad = gelu.backward(z_grad)

        torch_sum = torch_out.sum()
        torch_sum.backward()
        torch_grad = torch_inputs.grad

        print("grad: ", grad)
        print("torch grad: ", torch_grad.data.numpy())

        np.testing.assert_allclose(grad, torch_grad.data.numpy(), 1e-7, 1e-6)

    def test_tensor_gelu(self):

        np.random.seed(1)
        inputs = np.linspace(-15, 15, 100)
        gelu = GELU()
        out = gelu(inputs)

        torch_inputs = torch.tensor(inputs, dtype=torch.float)
        torch_inputs.requires_grad = True
        torch_gelu = nn.GELU()
        torch_out = torch_gelu.forward(torch_inputs)

        np.testing.assert_allclose(out, torch_out.detach().numpy(), 1e-7, 1e-6)

        z_grad = np.ones_like(inputs)
        grad = gelu.backward(z_grad)

        torch_sum = torch_out.sum()
        torch_sum.backward()
        torch_grad = torch_inputs.grad

        np.testing.assert_allclose(grad, torch_grad.data.numpy(), 1e-7, 1e-6)


