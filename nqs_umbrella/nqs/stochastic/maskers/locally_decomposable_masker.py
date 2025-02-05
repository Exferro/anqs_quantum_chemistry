import torch as pt

import os

from typing import Tuple

from ...base.constants import BASE_INT_TYPE

from .abstract_masker import AbstractMasker
from ..symmetries.abstract_locally_decomposable_symmetry import AbstractLocallyDecomposableSymmetry
from ..symmetries.abstract_additive_symmetry import AbstractAdditiveSymmetry
from ..symmetries.abstract_multiplicative_symmetry import AbstractMultiplicativeSymmetry

from ...infrastructure import create_dir


class LocallyDecomposableMasker(AbstractMasker):
    def __init__(self,
                 *args,
                 symmetries: Tuple[AbstractLocallyDecomposableSymmetry] = None,
                 **kwargs):
        super(LocallyDecomposableMasker, self).__init__(*args, **kwargs)
        for symmetry in symmetries:
            assert isinstance(symmetry, AbstractLocallyDecomposableSymmetry)
        self.symmetries = symmetries
        self.sym_num = len(self.symmetries)

        self.memo_size = 1
        self.is_multiplicative = pt.zeros(self.sym_num, dtype=pt.bool, device=self.device)
        self.bases = pt.zeros(self.sym_num, dtype=BASE_INT_TYPE, device=self.device)
        self.ref_acc_eigs = pt.zeros(self.sym_num, dtype=BASE_INT_TYPE, device=self.device)

        self.acc_eig2ordinal_mul_consts = pt.zeros(self.sym_num, dtype=BASE_INT_TYPE, device=self.device)
        self.acc_eig2ordinal_add_consts = pt.zeros(self.sym_num, dtype=BASE_INT_TYPE, device=self.device)
        self.acc_eig2ordinal_div_consts = pt.zeros(self.sym_num, dtype=BASE_INT_TYPE, device=self.device)

        self.local_eigs = pt.zeros((self.qubit_num, 2, self.sym_num), dtype=self.idx_dtype, device=self.device)
        self.min_bounds = pt.zeros((self.qubit_num + 1, self.sym_num), dtype=self.idx_dtype, device=self.device)
        self.max_bounds = pt.zeros((self.qubit_num + 1, self.sym_num), dtype=self.idx_dtype, device=self.device)
        for sym_idx, sym in enumerate(self.symmetries):
            if isinstance(sym, AbstractMultiplicativeSymmetry):
                self.is_multiplicative[sym_idx] = True

            self.bases[sym_idx] = self.memo_size
            self.memo_size *= sym.spectrum_size
            self.ref_acc_eigs[sym_idx] = sym.ref_eig

            self.acc_eig2ordinal_mul_consts[sym_idx] = sym.acc_eig2ordinal_mul_const
            self.acc_eig2ordinal_add_consts[sym_idx] = sym.acc_eig2ordinal_add_const
            self.acc_eig2ordinal_div_consts[sym_idx] = sym.acc_eig2ordinal_div_const

            self.min_bounds[0, sym_idx] = sym.min_acc_eig(0)
            self.max_bounds[0, sym_idx] = sym.max_acc_eig(0)
            for qubit_idx in range(self.qubit_num):
                self.local_eigs[qubit_idx, :, sym_idx] = sym.compute_part_eig(qubit_idx,
                                                                              pt.tensor([0, 1],
                                                                                        dtype=self.idx_dtype,
                                                                                        device=self.device))
                self.min_bounds[qubit_idx + 1, sym_idx] = sym.min_acc_eig(qubit_idx + 1)
                self.max_bounds[qubit_idx + 1, sym_idx] = sym.max_acc_eig(qubit_idx + 1)

        self.memo = pt.zeros((self.qubit_num + 1, self.memo_size), dtype=pt.bool, device=self.device)
        memos_dir = create_dir(os.path.join(self.parent_dir, 'memos'))
        self.memo_filename = os.path.join(memos_dir, f'{self.perm_type}_{self.memo.shape}_{self.ref_acc_eigs.detach().cpu()}')
        self.init_memo()

    def acc_eigs2memo_idx(self,
                          acc_eigs: pt.Tensor = None) -> pt.Tensor:
        if not pt.is_tensor(acc_eigs):
            acc_eigs = pt.tensor(acc_eigs, dtype=self.idx_dtype, device=self.device)

        memo_idx = (acc_eigs * self.acc_eig2ordinal_mul_consts + self.acc_eig2ordinal_add_consts) // self.acc_eig2ordinal_div_consts
        return (memo_idx * self.bases).sum(dim=-1)

    def memo_idx2acc_eigs(self,
                          memo_idx: pt.Tensor = None) -> pt.Tensor:
        if not pt.is_tensor(memo_idx):
            memo_idx = pt.tensor(memo_idx, dtype=self.idx_dtype, device=self.device)
        acc_eigs = []
        for sym_idx in range(self.sym_num - 1, -1, -1):
            cur_pos_ordinal = memo_idx // self.bases[sym_idx]
            memo_idx -= self.bases[sym_idx] * cur_pos_ordinal
            acc_eigs.append(self.symmetries[sym_idx].ordinal2acc_eig(cur_pos_ordinal))

        return pt.stack(acc_eigs[::-1], dim=1)

    def update_acc_eigs(self,
                        qubit_idx: int = None,
                        base_vec: pt.Tensor = None,
                        acc_eigs: pt.Tensor = None) -> pt.Tensor:

        return pt.where(pt.unsqueeze(self.is_multiplicative, dim=0),
                        acc_eigs * self.local_eigs[qubit_idx, base_vec, :],
                        acc_eigs + self.local_eigs[qubit_idx, base_vec, :])

    def acc_eigs_bound_check(self,
                             qubits_seen: int = None,
                             acc_eigs: pt.Tensor = None):

        return pt.all(pt.logical_and(pt.greater_equal(acc_eigs, self.min_bounds[qubits_seen, :]),
                                     pt.less_equal(acc_eigs, self.max_bounds[qubits_seen, :])), dim=-1)

    def acc_eigs2outcomes_masks(self,
                                qubits_seen: int = None,
                                acc_eigs: pt.Tensor = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        zero_outcome_eigs = self.update_acc_eigs(qubit_idx=qubits_seen,
                                                 base_vec=pt.tensor([0],
                                                                    dtype=self.idx_dtype,
                                                                    device=self.device),
                                                 acc_eigs=acc_eigs)
        zero_outcome_bound_mask = self.acc_eigs_bound_check(qubits_seen + 1, zero_outcome_eigs)
        zero_outcome_memo_idx = self.acc_eigs2memo_idx(zero_outcome_eigs)
        is_zero_outcome_physical = pt.zeros_like(zero_outcome_memo_idx, dtype=pt.bool)
        is_zero_outcome_physical[zero_outcome_bound_mask] = self.memo[
            qubits_seen + 1, zero_outcome_memo_idx[zero_outcome_bound_mask]]

        one_outcome_eigs = self.update_acc_eigs(qubit_idx=qubits_seen,
                                                base_vec=pt.tensor([1],
                                                                   dtype=self.idx_dtype,
                                                                   device=self.device),
                                                acc_eigs=acc_eigs)
        one_outcome_bound_mask = self.acc_eigs_bound_check(qubits_seen + 1, one_outcome_eigs)
        one_outcome_memo_idx = self.acc_eigs2memo_idx(one_outcome_eigs)
        is_one_outcome_physical = pt.zeros_like(one_outcome_memo_idx, dtype=pt.bool)
        is_one_outcome_physical[one_outcome_bound_mask] = self.memo[
            qubits_seen + 1, one_outcome_memo_idx[one_outcome_bound_mask]]

        return zero_outcome_eigs, is_zero_outcome_physical, one_outcome_eigs, is_one_outcome_physical

    def init_memo(self):
        if os.path.exists(self.memo_filename):
            self.memo = pt.load(self.memo_filename).to(self.device)
        else:
            all_memo_indices = pt.arange(self.memo_size, dtype=self.idx_dtype, device=self.device)
            last_qubit_eigs = self.memo_idx2acc_eigs(all_memo_indices)
            self.memo[self.qubit_num, :] = (last_qubit_eigs == self.ref_acc_eigs).all(dim=-1)
            for qubits_seen in range(self.qubit_num - 1, -1, -1):
                cur_acc_eigs = self.memo_idx2acc_eigs(pt.arange(self.memo_size, dtype=self.idx_dtype, device=self.device))
                cur_acc_eigs_bound_mask = self.acc_eigs_bound_check(qubits_seen, cur_acc_eigs)
                _, is_zero_outcome_physical, _, is_one_outcome_physical = self.acc_eigs2outcomes_masks(qubits_seen=qubits_seen,
                                                                                                 acc_eigs=cur_acc_eigs)

                self.memo[qubits_seen, :] = pt.logical_and(cur_acc_eigs_bound_mask,
                                                           pt.logical_or(is_zero_outcome_physical,
                                                                         is_one_outcome_physical))
            pt.save(self.memo.cpu(), self.memo_filename)

    def mask(self, base_vec: pt.Tensor) -> pt.Tensor:
        acc_eigs = []
        for sym_idx in range(self.sym_num):
            acc_eigs.append(self.symmetries[sym_idx].compute_acc_eig(base_vec))
        acc_eigs = pt.stack(acc_eigs, dim=-1)

        return self.memo[base_vec.shape[-1], self.acc_eigs2memo_idx(acc_eigs)]

    def compute_rolling_acc_eigs(self, base_vec: pt.Tensor = None) -> Tuple[pt.Tensor]:
        start_acc_eigs = []
        for sym_idx in range(self.sym_num):
            start_acc_eigs.append(pt.zeros(base_vec.shape[:-1],
                                           dtype=self.idx_dtype,
                                           device=self.device) + self.symmetries[sym_idx].start_eig)
        start_acc_eigs = pt.stack(start_acc_eigs, dim=-1)
        rolling_acc_eigs = [start_acc_eigs]
        for qubits_seen in range(base_vec.shape[-1]):
            rolling_acc_eigs.append(self.update_acc_eigs(qubits_seen, base_vec[..., qubits_seen], rolling_acc_eigs[qubits_seen]))

        return tuple(rolling_acc_eigs)

    def compute_rolling_outcome_masks(self, base_vec: pt.Tensor = None) -> Tuple[pt.Tensor]:
        rolling_acc_eigs = self.compute_rolling_acc_eigs(base_vec)
        rolling_outcome_masks = []
        for qubits_seen, acc_eigs in enumerate(rolling_acc_eigs[:-1]):
            _, is_zero_outcome_physical, _, is_one_outcome_physical = self.acc_eigs2outcomes_masks(qubits_seen=qubits_seen,
                                                                                                   acc_eigs=acc_eigs)
            rolling_outcome_masks.append(pt.stack((is_zero_outcome_physical, is_one_outcome_physical), dim=-1))

        return tuple(rolling_outcome_masks)


