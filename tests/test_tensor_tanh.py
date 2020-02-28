
from tinynet.tensor import Tensor
from tinynet.layers import Linear
from tinynet.tanh import Tanh
import unittest


class TestTensorTanh(unittest.TestCase):

    def test_tensor_tanh(self):

        tanh = Tanh()
        inputs = Tensor([[0, 0, 0], [100000, -100000, 10000]])
        outputs = tanh(inputs)
        print(outputs)
        grad = Tensor([[1, 1, 1], [1, 1, 1]])
        grad = tanh.backward(grad)
        print("backward", grad)