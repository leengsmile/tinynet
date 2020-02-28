
from numpy import ndarray as Tensor

# """
# Tensor
# """
#
#
# from typing import Union
# import numpy as np
#
#
# dtype = np.float32
# ndarray = np.ndarray
#
#
# ArrayType = Union[float, list, np.ndarray]
#
#
# def ensure_ndarray(data: ArrayType) -> np.ndarray:
#     if isinstance(data, np.ndarray):
#         return data
#     else:
#         return np.asarray(data)
#
#
# TensorLike = Union['Tensor', float, np.ndarray]
#
#
# def ensure_tensor(data: TensorLike) -> 'Tensor':
#     if isinstance(data, Tensor):
#         return data
#     else:
#         return Tensor(data)
#
#
# class Tensor:
#
#     def __init__(self, data: ArrayType, requires_grad: bool = False) -> None:
#         self.data = ensure_ndarray(data)
#         self.requires_grad = requires_grad
#         self.backward = None
#
#         if self.requires_grad:
#             self.backward = np.zeros_like(self.data)
#         self.shape = self.data.shape
#
#     def __str__(self) -> str:
#         return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
#
#     def __add__(self, other: TensorLike) -> 'Tensor':
#         return _add(self, ensure_tensor(other))
#
#     def __radd__(self, other: TensorLike) -> 'Tensor':
#         return _add(ensure_tensor(other), self)
#
#     def __sub__(self, other: TensorLike) -> 'Tensor':
#         return _sub(self, ensure_tensor(other))
#
#     def __rsub__(self, other: TensorLike) -> 'Tensor':
#         return _sub(ensure_tensor(other), self)
#
#     def __mul__(self, other: TensorLike) -> 'Tensor':
#         return _mul(self, ensure_tensor(other))
#
#     def __rmul__(self, other: TensorLike) -> 'Tensor':
#         return _mul(ensure_tensor(other), self)
#
#     def __matmul__(self, other: 'Tensor') -> 'Tensor':
#         return _matmul(self, other)
#
#     def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
#         return _matmul(other, self)
#
#     def __pos__(self):
#         return _pos(self)
#
#     def __neg__(self):
#         return _neg(self)
#
#     def __getitem__(self, idx: int) -> 'Tensor':
#         data = self.data[idx]
#         requires_grad = self.requires_grad
#         return Tensor(data, requires_grad)
#
#     def transpose(self):
#         data = self.data.T
#         requires_grad = self.requires_grad
#         return Tensor(data, requires_grad)
#
#     @property
#     def T(self):
#         return self.transpose()
#
#     def sum(self, axis: int = 0):
#         data = self.data.sum(axis=axis)
#         requires_grad = self.requires_grad
#         return Tensor(data, requires_grad)
#
#
# def _add(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data + t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     return Tensor(data, requires_grad)
#
#
# def _sub(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data - t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     return Tensor(data, requires_grad)
#
#
# def _mul(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data * t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     return Tensor(data, requires_grad)
#
#
# def _div(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data / t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     return Tensor(data, requires_grad)
#
#
# def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
#     data = t1.data @ t2.data
#     requires_grad = t1.requires_grad or t2.requires_grad
#     return Tensor(data, requires_grad)
#
#
# def _pos(t: Tensor) -> Tensor:
#     data = t.data
#     requires_grad = t.requires_grad
#     return Tensor(data, requires_grad)
#
#
# def _neg(t: Tensor) -> Tensor:
#     data = -t.data
#     requires_grad = t.requires_grad
#     return Tensor(data, requires_grad)
