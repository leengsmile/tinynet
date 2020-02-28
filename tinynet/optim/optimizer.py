from tinynet.tensor import Tensor


class Optimizer:

    def step(self, grad: Tensor):

        raise NotImplementedError
