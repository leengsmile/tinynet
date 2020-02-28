from typing import NamedTuple

import numpy as np
from .tensor import Tensor


class Batch(NamedTuple):
    inputs: Tensor
    targets: Tensor


class Dataset:

    def __iter__(self) -> 'Dataset':
        raise NotImplementedError

    def __next__(self) -> Batch:
        raise NotImplementedError


class BatchDataset(Dataset):
    def __init__(self, inputs: Tensor, targets: Tensor,
                 batch_size: int = 1, shuffle: bool = True) -> None:

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inputs = inputs
        self.targets = targets

    def __iter__(self) -> 'BatchDataset':
        self.starts = np.arange(0, len(self.inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.starts)
        self.start_idx = 0
        return self

    def __next__(self) -> Batch:

        if self.start_idx < len(self.starts):
            start = self.starts[self.start_idx]
            end = start + self.batch_size

            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            batch = Batch(batch_inputs, batch_targets)
            self.start_idx += 1
            return batch
        else:
            raise StopIteration

