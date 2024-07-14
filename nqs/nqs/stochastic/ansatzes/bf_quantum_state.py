import torch as pt
from torch import nn

from ...base.abstract_quantum_state import AbstractQuantumState

from typing import Tuple


class BFQuantumState(AbstractQuantumState):
    def __init__(self,
                 *args,
                 state_vec: pt.Tensor = None,
                 **kwargs):
        super(BFQuantumState, self).__init__(*args, **kwargs)

        if state_vec is not None:
            assert state_vec.shape == (self.dim, )
            assert state_vec.dtype == self.cdtype
            self.state_vec = state_vec
        else:
            self.state_vec = pt.randn((self.dim, ),
                                      dtype=self.cdtype,
                                      device=self.device)
        self.state_vec = pt.where(self.compute_phys_mask(pt.arange(2 ** self.qubit_num)),
                                  self.state_vec,
                                  pt.zeros_like(self.state_vec))
        self.state_vec = nn.Parameter(self.state_vec)

    def amplitude(self, base_idx: pt.Tensor) -> pt.Tensor:
        return self.state_vec[base_idx]

    def sample(self, sample_num: int) -> pt.Tensor:
        probs = pt.div(pt.conj(self.state_vec) * self.state_vec, self.norm() ** 2)
        sampled_indices = pt.multinomial(probs.real, sample_num, replacement=True)

        return sampled_indices

    def sample_stats(self, sample_num: int) -> Tuple[pt.LongTensor, pt.DoubleTensor]:
        sampled_indices = self.sample(sample_num=sample_num)
        unq_indices, freqs = pt.unique(sampled_indices, return_counts=True)

        return unq_indices, freqs.type(self.cdtype) / sample_num
