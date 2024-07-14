import torch as pt

from typing import Tuple

from .hilbert_space import HilbertSpace
from .abstract_hilbert_space_object import AbstractHilbertSpaceObject

from ..stochastic.maskers import LocallyDecomposableMasker

from ..infrastructure.nested_data import Config


class QubitGroupingConfig(Config):
    FIELDS = (
        'type',
        'qubit_per_qudit',
    )

    def __init__(self,
                 *args,
                 type: str = 'uniform',
                 qubit_per_qudit: int = 6,
                 **kwargs):
        self.type = type
        self.qubit_per_qudit = qubit_per_qudit

        super().__init__(*args, **kwargs)


class QubitGrouping(AbstractHilbertSpaceObject):
    def __init__(self,
                 *args,
                 qudit_starts: Tuple[int] = None,
                 qudit_ends: Tuple[int] = None,
                 masker: LocallyDecomposableMasker = None,
                 **kwargs):
        super(QubitGrouping, self).__init__(*args, **kwargs)
        assert len(qudit_starts) == len(qudit_ends)
        for qudit_idx in range(len(qudit_starts)):
            assert 0 <= qudit_starts[qudit_idx]
            assert qudit_starts[qudit_idx] <= self.qubit_num - 1

            assert 1 <= qudit_ends[qudit_idx]
            assert qudit_ends[qudit_idx] <= self.qubit_num

            assert qudit_starts[qudit_idx] < qudit_ends[qudit_idx]

            if (0 < qudit_idx) and (qudit_idx < len(qudit_starts) - 1):
                assert qudit_starts[qudit_idx - 1] < qudit_starts[qudit_idx]
                assert qudit_starts[qudit_idx] < qudit_starts[qudit_idx + 1]
                assert qudit_ends[qudit_idx - 1] < qudit_ends[qudit_idx]
                assert qudit_ends[qudit_idx] < qudit_ends[qudit_idx + 1]

        self.qudit_num = len(qudit_starts)
        self.qudit_starts = qudit_starts
        self.qudit_ends = qudit_ends

        self.qubits_per_qudit = tuple(self.qudit_ends[qudit_idx] - self.qudit_starts[qudit_idx]
                                      for qudit_idx in range(self.qudit_num))

        self.qudit_dims = tuple(2**self.qubits_per_qudit[qudit_idx] for qudit_idx in range(self.qudit_num))
        self.qudit_dims = pt.tensor(self.qudit_dims, device=self.device)

        self.qubit_idx2qudit_two_power = ()
        self.qubit_idx2qudit_idx = ()
        for qudit_idx in range(self.qudit_num):
            self.qubit_idx2qudit_two_power += tuple(2**qubit_idx for qubit_idx in range(self.qubits_per_qudit[qudit_idx]))
            self.qubit_idx2qudit_idx += tuple([qudit_idx] * self.qubits_per_qudit[qudit_idx])
        self.qubit_idx2qudit_two_power = pt.tensor(self.qubit_idx2qudit_two_power,
                                                   dtype=self.idx_dtype,
                                                   device=self.device)
        self.qubit_idx2qudit_idx = pt.tensor(self.qubit_idx2qudit_idx,
                                             dtype=self.idx_dtype,
                                             device=self.device)

        self.qudit_idx2local_base_vecs = []
        for qudit_idx in range(self.qudit_num):
            self.qudit_idx2local_base_vecs.append(self.qudit2base_vec(pt.arange(self.qudit_dims[qudit_idx],
                                                                                dtype=self.idx_dtype,
                                                                                device=self.device),
                                                                      qubit_per_qudit=self.qubits_per_qudit[qudit_idx]))
        self.qudit_idx2local_base_vecs = tuple(self.qudit_idx2local_base_vecs)

        self.masker = masker
        self.qudit_idx2local_eigs = []
        for qudit_idx in range(self.qudit_num):
            acc_eigs = []
            for sym_idx in range(self.masker.sym_num):
                acc_eigs.append(pt.zeros(self.qudit_idx2local_base_vecs[qudit_idx].shape[:-1],
                                         dtype=self.idx_dtype,
                                         device=self.device) + self.masker.symmetries[sym_idx].start_eig)
            acc_eigs = pt.stack(acc_eigs, dim=-1)
            for qubit_idx in range(self.qubits_per_qudit[qudit_idx]):
                acc_eigs = self.masker.update_acc_eigs(qubit_idx=self.qudit_starts[qudit_idx] + qubit_idx,
                                                       base_vec=self.qudit_idx2local_base_vecs[qudit_idx][..., qubit_idx],
                                                       acc_eigs=acc_eigs)
            self.qudit_idx2local_eigs.append(acc_eigs)

        self.qudit_idx2memo_idx_mul_table = []
        self.qudit_idx2cont_mask_mul_table = []
        self.memo_idx_arange = pt.arange(self.masker.memo_size, device=self.device)
        self.memo_idx_acc_eigs = self.masker.memo_idx2acc_eigs(self.memo_idx_arange)
        for qudit_idx in range(self.qudit_num):
            new_acc_eigs, cont_mask = self.qudit_acc_eigs2continuation_mask(qudit_idx=qudit_idx,
                                                                            qudit_acc_eigs=self.memo_idx_acc_eigs)
            self.qudit_idx2memo_idx_mul_table.append(pt.reshape(self.masker.acc_eigs2memo_idx(new_acc_eigs),
                                                                (-1, self.qudit_dims[qudit_idx])))
            self.qudit_idx2cont_mask_mul_table.append(pt.reshape(cont_mask, (-1, self.qudit_dims[qudit_idx])))


    @classmethod
    def create(cls,
               config: QubitGroupingConfig = None,
               hs: HilbertSpace = None,
               masker: LocallyDecomposableMasker = None):
        config = config if config is not None else QubitGroupingConfig()
        if config.type == 'uniform':
            qubit_per_qudit = config.qubit_per_qudit
            qudit_num = hs.qubit_num // qubit_per_qudit
            if hs.qubit_num % qubit_per_qudit:
                qudit_num += 1
            qudit_starts = tuple(qudit_idx * qubit_per_qudit for qudit_idx in range(qudit_num))
            qudit_ends = qudit_starts[1:] + (hs.qubit_num,)

            return QubitGrouping(hilbert_space=hs,
                                 qudit_starts=qudit_starts,
                                 qudit_ends=qudit_ends,
                                 masker=masker)

    @staticmethod
    def qudit2base_vec(qudit: pt.Tensor,
                       qubit_per_qudit: int = None) -> pt.Tensor:
        shifts = pt.arange(0,
                           qubit_per_qudit,
                           dtype=qudit.dtype,
                           device=qudit.device)

        return (qudit.reshape(-1, 1) >> shifts).remainder_(2)
    
    @staticmethod
    def base_vec2qudit(base_vec: pt.Tensor,
                       qubit_per_qudit: int = None) -> pt.Tensor:
        assert base_vec.shape[-1] == qubit_per_qudit
        return pt.sum(base_vec * (2 ** pt.arange(qubit_per_qudit,
                                                 dtype=base_vec.dtype,
                                                 device=base_vec.device)), dim=-1)
    
    def base_vec2qudit_base_vec(self, base_vec: pt.Tensor) -> pt.Tensor:
        return pt.scatter_add(pt.zeros((base_vec.shape[0], self.qudit_num),
                                       dtype=self.idx_dtype,
                                       device=self.device),
                              dim=1,
                              index=pt.broadcast_to(self.qubit_idx2qudit_idx, base_vec.shape),
                              src=base_vec * self.qubit_idx2qudit_two_power)

    def base_vec2qudit_rolling_acc_eigs(self, base_vec: pt.Tensor) -> Tuple[pt.Tensor]:
        """
        A function to calculate accumulated eigenvalues for partial basis vectors, where the separation
        stems from qudit boundaries
        """
        full_rolling_acc_eigs = self.masker.compute_rolling_acc_eigs(base_vec)
        qudit_rolling_acc_eigs = (full_rolling_acc_eigs[0],) + tuple(full_rolling_acc_eigs[self.qudit_ends[qudit_idx]]
                                                                     for qudit_idx in range(self.qudit_num))

        return qudit_rolling_acc_eigs

    def qudit_acc_eigs2continuation_mask(self,
                                         qudit_idx: int = None,
                                         qudit_acc_eigs: pt.Tensor = None) -> [pt.Tensor, pt.Tensor]:
        """
        qudit_acc_eigs is a (batched) vector of eigenvalues accumulated so far by qudit_idx.
        We calculate accumulated eigenvalues for all possible continuations of the vector, and then
        calculate physicality masks for them.

        In principle, masker should be able to do something similar itself, but in practice masker doesn't know
        anything about qudits. Hence, here everything becomes a bit complicated and we basically copypaste some
        typical masker actions (e.g. borders check).
        """
        broadcast_qudit_local_eigs = pt.tile(self.qudit_idx2local_eigs[qudit_idx], (qudit_acc_eigs.shape[0], 1))

        qudit_acc_eigs = pt.unsqueeze(qudit_acc_eigs, dim=1)
        qudit_acc_eigs = pt.tile(qudit_acc_eigs, (1, self.qudit_dims[qudit_idx], 1))
        qudit_acc_eigs = pt.reshape(qudit_acc_eigs, (self.qudit_dims[qudit_idx] * qudit_acc_eigs.shape[0], -1))

        new_acc_eigs = pt.where(pt.unsqueeze(self.masker.is_multiplicative, dim=0),
                                qudit_acc_eigs * broadcast_qudit_local_eigs,
                                qudit_acc_eigs + broadcast_qudit_local_eigs)

        bound_mask = self.masker.acc_eigs_bound_check(self.qudit_ends[qudit_idx], new_acc_eigs)
        memo_idx = self.masker.acc_eigs2memo_idx(new_acc_eigs)
        #print(memo_idx.shape)
        #print(pt.unique(memo_idx).shape)
        #print()
        mask = pt.zeros_like(memo_idx, dtype=pt.bool)
        mask[bound_mask] = self.masker.memo[self.qudit_ends[qudit_idx], memo_idx[bound_mask]]

        return new_acc_eigs, mask

    def base_vec2qudit_continuation_masks(self,
                                          base_vec: pt.Tensor = None) -> Tuple[pt.Tensor]:
        """
        Effectively a wrapper around the above function to make it a) applicable to base_vecs; b) rolling.
        """
        qudit_rolling_acc_eigs = self.base_vec2qudit_rolling_acc_eigs(base_vec)
        qudit_rolling_memo_idx = [self.masker.acc_eigs2memo_idx(acc_eigs) for acc_eigs in qudit_rolling_acc_eigs]
        qudit_continuation_masks = []
        for qudit_idx in range(self.qudit_num):
            # _, mask = self.qudit_acc_eigs2continuation_mask(qudit_idx=qudit_idx,
            #                                                 qudit_acc_eigs=qudit_rolling_acc_eigs[qudit_idx])
            mask = self.qudit_idx2cont_mask_mul_table[qudit_idx][qudit_rolling_memo_idx[qudit_idx]]
            qudit_continuation_masks.append(pt.reshape(mask, (-1, )))

        return tuple(qudit_continuation_masks)

