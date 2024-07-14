import torch as pt

from ...base.constants import BASE_INT_TYPE

from .abstract_additive_symmetry import AbstractAdditiveSymmetry


class ParticleNumberSymmetry(AbstractAdditiveSymmetry):
    def __init__(self,
                 *args,
                 particle_num: int = None,
                 **kwargs):
        super(ParticleNumberSymmetry, self).__init__(*args, **kwargs)
        assert particle_num is not None
        assert (0 <= particle_num) and (particle_num <= self.qubit_num)
        self.particle_num = particle_num
        
    @property
    def spectrum_size(self):
        return self.qubit_num + 1

    def min_acc_eig(self, qubits_seen: int = None):
        return 0

    def max_acc_eig(self, qubits_seen: int = None):
        return qubits_seen

    @property
    def ref_eig(self):
        return self.particle_num

    def compute_part_eig(self,
                         qubit_idx: int = None,
                         base_vec: pt.Tensor = None) -> pt.Tensor:
        assert (0 <= qubit_idx) and (qubit_idx <= self.qubit_num)
        assert base_vec.dtype == BASE_INT_TYPE
        
        return base_vec

    def update_acc_eig(self,
                       qubits_seen: int = None,
                       base_vec: pt.Tensor = None,
                       acc_eig: pt.Tensor = None) -> pt.Tensor:
        assert (0 <= qubits_seen) and (qubits_seen <= self.qubit_num)
        assert base_vec.dtype == BASE_INT_TYPE
        assert acc_eig.dtype == BASE_INT_TYPE

        return acc_eig + self.compute_part_eig(qubits_seen, base_vec)

    @property
    def acc_eig2ordinal_mul_const(self):
        return 1

    @property
    def acc_eig2ordinal_add_const(self):
        return 0

    @property
    def acc_eig2ordinal_div_const(self):
        return 1
