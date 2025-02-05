import torch as pt

from abc import ABC

from typing import Tuple

from .hilbert_space import HilbertSpace


class AbstractHilbertSpaceObject(ABC):
    def __init__(self,
                 *,
                 hilbert_space: HilbertSpace = None):
        super(AbstractHilbertSpaceObject, self).__init__()
        assert hilbert_space is not None
        self.hilbert_space = hilbert_space

    @property
    def device(self):
        return self.hilbert_space.device

    @property
    def qubit_num(self) -> int:
        return self.hilbert_space.qubit_num

    @property
    def rdtype(self):
        return self.hilbert_space.rdtype

    @property
    def cdtype(self):
        return self.hilbert_space.cdtype

    @property
    def idx_dtype(self):
        return self.hilbert_space.idx_dtype

    @property
    def parent_dir(self):
        return self.hilbert_space.parent_dir

    @property
    def rng_seed(self):
        return self.hilbert_space.rng_seed

    @property
    def rng(self):
        return self.hilbert_space.rng

    @property
    def perm_type(self):
        return self.hilbert_space.perm_type

    @property
    def perm(self):
        return self.hilbert_space.perm

    @property
    def inv_perm(self):
        return self.hilbert_space.inv_perm

    # noinspection DuplicatedCode
    def base_idx2base_vec(self, base_idx: pt.Tensor) -> pt.Tensor:
        return self.hilbert_space.base_idx2base_vec(base_idx=base_idx)

    # noinspection DuplicatedCode
    def base_vec2base_idx(self, base_vec: pt.Tensor) -> pt.Tensor:
        return self.hilbert_space.base_vec2base_idx(base_vec=base_vec)

    def popcount(self, base_idx: pt.Tensor) -> pt.Tensor:
        return self.hilbert_space.popcount(base_idx).to(self.device)

    def popcount_(self, base_idx: pt.Tensor) -> pt.Tensor:
        return self.hilbert_space.popcount_(base_idx).to(self.device)

    def old_popcount(self, base_idx: pt.Tensor) -> pt.Tensor:
        return self.hilbert_space.old_popcount(base_idx).to(self.device)

    def sort_base_idx(self, base_idx: pt.Tensor, descending: bool = False) -> pt.Tensor:
        return self.hilbert_space.sort_base_idx(base_idx=base_idx, descending=descending)

    def find_a_in_b(self, a: pt.Tensor = None, b: pt.Tensor = None) -> Tuple[pt.Tensor, pt.Tensor]:
        return self.hilbert_space.find_a_in_b(a=a, b=b)
