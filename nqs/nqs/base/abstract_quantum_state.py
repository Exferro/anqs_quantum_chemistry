import torch as pt
from torch import nn
import numpy as np

from abc import abstractmethod

from .abstract_hilbert_space_object import AbstractHilbertSpaceObject

from typing import Tuple


class AbstractQuantumState(AbstractHilbertSpaceObject, nn.Module):
    # we pass the number of qubits N
    def __init__(self, *args, **kwargs):
        super(AbstractQuantumState, self).__init__(*args, **kwargs)
        self.shapes = []
        self.splits = []

    def forward(self, base_idx: pt.Tensor) -> pt.Tensor:
        return self.amplitude(base_idx=base_idx)

    # measuring amplitude with respect to some basis vector
    @abstractmethod
    def amplitude(self, base_idx: pt.Tensor) -> pt.Tensor:
        ...

    @abstractmethod
    def sample_stats(self, sample_num: int) -> Tuple[pt.LongTensor, pt.Tensor]:
        ...

    def init_shapes_and_splits(self):
        for param in self.parameters():
            self.shapes.append(param.shape)
            self.splits.append(np.prod(param.shape))

    def get_param_vec(self):
        param_vec = []
        for param in self.parameters():
            param_vec.append(pt.reshape(param.data, (-1, )))

        return pt.cat(param_vec)

    def set_param_vec(self, param_vec: pt.Tensor):
        param_vec = pt.split(param_vec, self.splits)
        for param_idx, param in enumerate(self.parameters()):
            param.data = pt.reshape(param_vec[param_idx], self.shapes[param_idx])

    def get_param_vec_grad(self):
        param_vec_grad = []
        for param in self.parameters():
            param_vec_grad.append(pt.reshape(param.grad.data, (-1, )))

        return pt.cat(param_vec_grad)

    def set_param_vec_grad(self, param_vec_grad: pt.Tensor):
        param_vec_grad = pt.split(param_vec_grad, self.splits)
        for param_idx, param in enumerate(self.parameters()):
            if param.grad is not None:
                param.grad.data = pt.reshape(param_vec_grad[param_idx], self.shapes[param_idx])
