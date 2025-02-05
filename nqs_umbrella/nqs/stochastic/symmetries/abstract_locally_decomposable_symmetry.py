import torch as pt

from abc import abstractmethod

from ...base.constants import BASE_INT_TYPE
from ...base.abstract_hilbert_space_object import AbstractHilbertSpaceObject


class AbstractLocallyDecomposableSymmetry(AbstractHilbertSpaceObject):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AbstractLocallyDecomposableSymmetry, self).__init__(*args, **kwargs)

    @property
    @abstractmethod
    def spectrum_size(self):
        ...

    @property
    @abstractmethod
    def start_eig(self):
        ...

    @property
    @abstractmethod
    def ref_eig(self):
        ...

    @abstractmethod
    def min_acc_eig(self, qubits_seen: int = None):
        ...

    @abstractmethod
    def max_acc_eig(self, qubits_seen: int = None):
        ...

    @abstractmethod
    def compute_part_eig(self,
                         qubit_idx: int = None,
                         base_vec: pt.Tensor = None) -> pt.Tensor:
        ...

    @abstractmethod
    def update_acc_eig(self,
                       qubits_seen: int = None,
                       base_vec: pt.Tensor = None,
                       acc_eig: pt.Tensor = None) -> pt.Tensor:
        ...

    @property
    @abstractmethod
    def acc_eig2ordinal_mul_const(self):
        ...

    @property
    @abstractmethod
    def acc_eig2ordinal_add_const(self):
        ...

    @property
    @abstractmethod
    def acc_eig2ordinal_div_const(self):
        ...

    def acc_eig2ordinal(self,
                        acc_eig: pt.Tensor = None) -> pt.Tensor:
        if not pt.is_tensor(acc_eig):
            acc_eig = pt.tensor(acc_eig, dtype=self.idx_dtype, device=self.device)
        assert acc_eig.dtype == BASE_INT_TYPE

        return (acc_eig * self.acc_eig2ordinal_mul_const + self.acc_eig2ordinal_add_const) // self.acc_eig2ordinal_div_const

    def ordinal2acc_eig(self,
                        ordinal: pt.Tensor = None) -> pt.Tensor:
        if not pt.is_tensor(ordinal):
            ordinal = pt.tensor(ordinal, dtype=self.idx_dtype, device=self.device)
        assert ordinal.dtype == BASE_INT_TYPE

        return (ordinal * self.acc_eig2ordinal_div_const - self.acc_eig2ordinal_add_const) // self.acc_eig2ordinal_mul_const

    def compute_acc_eig(self,
                        base_vec):
        acc_eig = pt.zeros(base_vec.shape[:-1],
                           dtype=BASE_INT_TYPE,
                           device=self.device) + self.start_eig
        for qubits_seen in range(base_vec.shape[-1]):
            acc_eig = self.update_acc_eig(qubits_seen=qubits_seen,
                                          base_vec=base_vec[..., qubits_seen],
                                          acc_eig=acc_eig)

        return acc_eig
