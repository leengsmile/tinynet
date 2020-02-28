from tinynet.tensor import Tensor
import unittest
import pytest


@pytest.mark.skip(reason="default Tensor has no constructor.")
class TestTensorInit(unittest.TestCase):

    def test_tensor_init(self):
        t = Tensor([1, 2, 3])
        print("tensor", t)

        assert t.data.tolist() == [1, 2, 3]
        assert t.grad is None


