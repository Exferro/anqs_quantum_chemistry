import torch as pt

from abc import abstractmethod

from ...base.constants import BASE_INT_TYPE

from .abstract_locally_decomposable_symmetry import AbstractLocallyDecomposableSymmetry


class AbstractMultiplicativeSymmetry(AbstractLocallyDecomposableSymmetry):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AbstractMultiplicativeSymmetry, self).__init__(*args, **kwargs)

    @property
    def start_eig(self):
        return 1

    def update_acc_eig(self,
                       qubits_seen: int = None,
                       base_vec: pt.Tensor = None,
                       acc_eig: pt.Tensor = None) -> pt.Tensor:
        assert (0 <= qubits_seen) and (qubits_seen <= self.qubit_num)
        assert base_vec.dtype == BASE_INT_TYPE
        assert acc_eig.dtype == BASE_INT_TYPE

        return acc_eig * self.compute_part_eig(qubits_seen, base_vec)
