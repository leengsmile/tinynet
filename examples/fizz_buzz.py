import numpy as np
np.random.seed(1)
from typing import List
from tinynet import Tensor
from tinynet import Linear, Tanh, Layer, Sequential
from tinynet import Module, CrossEntropy, MSE
from tinynet import SGD, Adagrad, Adam
from tinynet import BatchDataset


def binary_encode(x: int, n: int = 10) -> List[int]:
    return [x >> i & 1 for i in range(n)]


def fizz_buzz_encode(x: int) -> int:

    if x % 15 == 0:
        return 3
        # return [0, 0, 0, 1]
    elif x % 5 == 0:
        return 2
        # return [0, 0, 1, 0]
    elif x % 3 == 0:
        return 1
        # return [0, 1, 0, 0]
    else:
        return 0
        # return [1, 0, 0, 0]


class FizzBuzzModel(Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.layers = Sequential([
            Linear(input_size, hidden_size),
            Tanh(),
            Linear(hidden_size, output_size)
        ])

    def __call__(self, inputs: Tensor) -> Tensor:
        z = self.layers(inputs)
        return z

    def backward(self, grad: Tensor) -> Tensor:
        return self.layers.backward(grad)


train_data = np.array([binary_encode(x) for x in range(101, 2**10)])
train_label = np.array([fizz_buzz_encode(x) for x in range(101, 2 ** 10)])

train_dataset = BatchDataset(train_data, train_label, batch_size=200, shuffle=True)

model = FizzBuzzModel(train_data.shape[1], 50, 4)
criterion = CrossEntropy()
# optimizer = SGD(model, lr=0.0005, momentum=0.2, nesterov=True)
optimizer = Adagrad(model, lr=0.5, weight_decay=0.0, eps=1e-10)

# optimizer = Adam(model, lr=0.001, betas=(0.9, 0.99))


def evaluate(model: Module):
    hit, total = 0, 0
    for x in range(1, 101):
        valid_data = np.array([binary_encode(x)])
        valid_out = model(valid_data)
        predicted_idx = valid_out.argmax(axis=1)[0]
        actual_idx = fizz_buzz_encode(x)
        if predicted_idx == actual_idx:
            hit += 1
        total += 1
    return hit / total


for epoch in range(5000):
    total_loss = 0.
    for batch_inputs, batch_targets in train_dataset:

        model.zero_grad()
        out = model(train_data)
        loss = criterion(out, train_label)
        grad = criterion.backward(out, train_label)
        model.backward(grad)
        optimizer.step(grad)
        total_loss += loss
    if (epoch + 1) % 100 == 0:
        accuracy = evaluate(model)
        # accuracy = 0
        print(f"epoch: {epoch+1}, loss: {total_loss / len(train_data)}, accuracy: {accuracy}")

valid_data = np.array([binary_encode(x) for x in range(1, 101)])
valid_label = np.array([fizz_buzz_encode(x) for x in range(1, 101)])
hit, total = 0, 0
for x in range(1, 101):
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    valid_data = np.array([binary_encode(x)])
    valid_out = model(valid_data)
    predicted_idx = valid_out.argmax(axis=1)[0]
    actual_idx = fizz_buzz_encode(x)
    if predicted_idx == actual_idx:
        hit += 1
    total += 1
    print(x, labels[actual_idx], labels[predicted_idx])
print(f"hit: {hit}, total: {total}")