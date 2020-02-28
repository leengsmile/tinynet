import numpy as np
from tinynet.data import BatchDataset
import unittest


class TestTensorDataset(unittest.TestCase):

    def test_tensor_dataset(self):
        inputs = np.random.randn(10, 2)
        targets = np.random.randint(0, 2, inputs.shape[0])

        batch_size = 2
        dataset = BatchDataset(inputs=inputs, targets=targets, batch_size=batch_size)

        for batch in dataset:
            print(batch)

    def test_tensor_dataset_shuffle(self):
        inputs = np.random.randn(10, 2)
        targets = np.random.randint(0, 2, inputs.shape[0])

        batch_size = 2
        dataset = BatchDataset(inputs=inputs, targets=targets, batch_size=batch_size, shuffle=True)

        for batch in dataset:
            print(batch)

    def test_tensor_dataset_iter(self):
        np.random.seed(1)
        inputs = np.random.randn(4, 2)
        targets = np.random.randint(0, 2, inputs.shape[0])

        batch_size = 2
        dataset = BatchDataset(inputs=inputs, targets=targets, batch_size=batch_size, shuffle=True)

        for epoch in range(2):
            for bi, batch in enumerate(dataset):
                print("- " * 10, epoch, bi, " -" * 10)
                print(epoch, batch)


