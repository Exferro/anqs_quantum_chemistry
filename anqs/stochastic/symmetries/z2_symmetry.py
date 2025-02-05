import torch as pt

from ...base.constants import BASE_INT_TYPE

from .abstract_multiplicative_symmetry import AbstractMultiplicativeSymmetry


class Z2Symmetry(AbstractMultiplicativeSymmetry):
    def __init__(self,
                 *args,
                 value: int = None,
                 pauli_z_positions: pt.Tensor = None,
                 **kwargs):
        super(Z2Symmetry, self).__init__(*args, **kwargs)
        assert value in (-1, 1, None)
        self.value = value
        self.pauli_z_positions = pauli_z_positions
        self.pauli_z_mask = pt.zeros(self.qubit_num, dtype=BASE_INT_TYPE, device=self.device)
        for pauli_z_pos in self.pauli_z_positions:
            assert (0 <= pauli_z_pos) and (pauli_z_pos <= self.qubit_num)
            self.pauli_z_mask[pauli_z_pos] = 1

    @property
    def spectrum_size(self):
        return 2

    @property
    def ref_eig(self):
        return self.value

    def min_acc_eig(self, qubits_seen: int = None):
        return -1

    def max_acc_eig(self, qubits_seen: int = None):
        return 1

    def compute_part_eig(self,
                         qubit_idx: int = None,
                         base_vec: pt.Tensor = None) -> pt.Tensor:
        assert (0 <= qubit_idx) and (qubit_idx <= self.qubit_num)
        assert base_vec.dtype == BASE_INT_TYPE

        return (-1)**(self.pauli_z_mask[qubit_idx] * base_vec)

    @property
    def acc_eig2ordinal_mul_const(self):
        return -1

    @property
    def acc_eig2ordinal_add_const(self):
        return 1

    @property
    def acc_eig2ordinal_div_const(self):
        return 2
