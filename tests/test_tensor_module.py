from tinynet.module import Module
from tinynet.layers import Linear
import unittest


class TestTensorModule(unittest.TestCase):

    def test_tensor_module(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(2, 3)


