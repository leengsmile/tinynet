from tinynet.tensor import Tensor
import unittest


class TestTensorMatMul(unittest.TestCase):

    def test_tensor_matmul(self):
        t1 = Tensor([1, 2, 3])
        t2 = t1 * 2
        assert t2.data.tolist() == [2, 4, 6]

        t3 = 2 * t1
        assert t3.data.tolist() == [2, 4, 6]

