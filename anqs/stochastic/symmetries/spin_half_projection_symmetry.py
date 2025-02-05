import torch as pt

from ...base.constants import BASE_INT_TYPE

from .abstract_additive_symmetry import AbstractAdditiveSymmetry


class SpinHalfProjectionSymmetry(AbstractAdditiveSymmetry):
    def __init__(self,
                 *args,
                 spin: int = None,
                 **kwargs):
        super(SpinHalfProjectionSymmetry, self).__init__(*args, **kwargs)
        assert spin is not None
        assert (-self.qubit_num <= spin) and (spin <= self.qubit_num)
        self.spin = spin

        self.min_acc_eigs = pt.zeros(self.qubit_num + 1, dtype=self.idx_dtype)
        self.max_acc_eigs = pt.zeros(self.qubit_num + 1, dtype=self.idx_dtype)
        for qubits_seen in range(1, self.qubit_num + 1):
            if (self.inv_perm[qubits_seen - 1] % 2) == 0:
                self.max_acc_eigs[qubits_seen] = self.max_acc_eigs[qubits_seen - 1] + 1
            else:
                self.max_acc_eigs[qubits_seen] = self.max_acc_eigs[qubits_seen - 1]

            if (self.inv_perm[qubits_seen - 1] % 2) == 1:
                self.min_acc_eigs[qubits_seen] = self.min_acc_eigs[qubits_seen - 1] - 1
            else:
                self.min_acc_eigs[qubits_seen] = self.min_acc_eigs[qubits_seen - 1]


    @property
    def spectrum_size(self):
        return (self.qubit_num + 1) // 2 + (self.qubit_num // 2) + 1

    def min_acc_eig(self, qubits_seen: int = None):
        return self.min_acc_eigs[qubits_seen]#-(qubits_seen // 2)

    def max_acc_eig(self, qubits_seen: int = None):
        return self.max_acc_eigs[qubits_seen]#(qubits_seen + 1) // 2

    @property
    def ref_eig(self):
        return self.spin

    def compute_part_eig(self,
                         qubit_idx: int = None,
                         base_vec: pt.Tensor = None) -> pt.Tensor:
        assert (0 <= qubit_idx) and (qubit_idx <= self.qubit_num)
        assert base_vec.dtype == BASE_INT_TYPE

        return base_vec * ((-1)**self.inv_perm[qubit_idx])

    @property
    def acc_eig2ordinal_mul_const(self):
        return 1

    @property
    def acc_eig2ordinal_add_const(self):
        return self.qubit_num // 2

    @property
    def acc_eig2ordinal_div_const(self):
        return 1
