
from tinynet.tensor import Tensor
from tinynet.layers import Linear
from tinynet.parameter import Parameter
import unittest


class TestTensorModule(unittest.TestCase):

    def test_tensor_linear(self):

        linear = Linear(2, 3)
        linear.weights = Parameter([[1, 1, 1], [2, 2, 2]])
        linear.bias = Parameter([3, 3, 3])
        # linear.params["w"] = Tensor([[1, 1, 1], [2, 2, 2]])
        # linear.params["b"] = Tensor([3, 3, 3])
        inputs = Tensor([[1, 1], [2, 2], [3, 3]])
        outputs = linear(inputs)
        print("linear", linear.params)
        print("output", outputs)

        assert outputs.data.tolist() == [[6, 6, 6], [9, 9, 9], [12, 12, 12]]

        grad = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        grad = linear.backward(grad)
        assert grad.data.tolist() == [[3, 6], [3, 6], [3, 6]]

        assert linear.weights.backward.tolist() == [[6, 6, 6], [6, 6, 6]]
        assert linear.bias.backward.tolist() == [3, 3, 3]

