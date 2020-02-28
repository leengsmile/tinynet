from tinynet.tensor import Tensor
import unittest


class TestTensorMatMul(unittest.TestCase):

    def test_tensor_matmul(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        t2 = Tensor([1, 1, 1])
        t3 = t1 @ t2

        assert t3.data.tolist() == [6, 15]
        print(t3)


