from abc import abstractmethod
import torch as pt
import numpy as np

from typing import Tuple, Union

from nqs.base.constants import NEGINF
from nqs.base.constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE

from nqs.base import AbstractQuantumState
from nqs.stochastic.maskers.locally_decomposable_masker import LocallyDecomposableMasker


from enum import Enum
MaskingMode = Enum('MaskingMode', 'logits part_base_vecs')


class AbstractANQS(AbstractQuantumState):
    def __init__(self,
                 *args,
                 masker: Union[None, LocallyDecomposableMasker] = None,
                 masking_mode: MaskingMode = None,
                 masking_depth: int = 2,
                 perseverant_sampling: Union[bool, str] = True,
                 split_size: int = np.inf,
                 **kwargs):
        super(AbstractANQS, self).__init__(*args, **kwargs)

        assert isinstance(masker, LocallyDecomposableMasker) or isinstance(masker, IdleMasker)
        self.masker = masker
        assert masking_mode in (MaskingMode.logits, MaskingMode.part_base_vecs, None)
        self.masking_mode = masking_mode
        self.masking_depth = masking_depth

        assert perseverant_sampling in (True, False, 'renorm')
        self.perseverant_sampling = perseverant_sampling

        self.split_size = split_size

    @property
    @abstractmethod
    def inp_dtype(self):
        ...

    def forward(self, x):
        return self.amplitude(x)

    def amplitude(self, base_idx: pt.Tensor) -> pt.Tensor:
        base_vec = self.base_idx2base_vec(base_idx).type(self.inp_dtype).to(self.device)
        return pt.exp(self.log_psi(base_vec))

    def phase(self, x):

        return self.log_psi(x).imag

    @abstractmethod
    def log_psi(self, x: pt.Tensor):
        ...

    def log_abs(self, x: pt.Tensor):

        return self.log_psi(x).real

    @abstractmethod
    def cond_log_psi(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:
        ...

    def cond_log_abs(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:

        return self.cond_log_psi(x, idx).real

    def mask_logits(self, logits: pt.Tensor = None, part_base_vec: pt.Tensor = None) -> pt.Tensor:
        if (self.masker is not None) and (self.masking_mode == MaskingMode.logits):
            shape = part_base_vec.shape
            if shape[-1] < self.qubit_num - self.masking_depth:
                zero_outcome = pt.cat((part_base_vec,
                                       pt.zeros((*shape[:-1], 1),
                                                dtype=part_base_vec.dtype,
                                                device=self.device)),
                                      dim=-1)
                one_outcome = pt.cat((part_base_vec,
                                       pt.ones((*shape[:-1], 1),
                                               dtype=part_base_vec.dtype,
                                               device=self.device)),
                                     dim=-1)
                zero_outcome = pt.unsqueeze(zero_outcome, dim=1)
                one_outcome = pt.unsqueeze(one_outcome, dim=1)
                outcome_stack = pt.cat((zero_outcome, one_outcome), dim=1)

                mask = self.masker.mask(outcome_stack)

                logits = self.mask_logits_given_mask(logits=logits, mask=mask)

        return logits

    def mask_logits_given_mask(self, logits: pt.Tensor = None, mask: pt.Tensor = None):
        if logits.dtype == BASE_REAL_TYPE:
            logits[~mask] = NEGINF.type(self.rdtype)
            logits = logits - 0.5 * pt.logsumexp(2 * logits, dim=-1, keepdim=True)
            logits = pt.nan_to_num(logits, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype))
        elif logits.dtype == BASE_COMPLEX_TYPE:
            logits.real[~mask] = NEGINF.type(self.rdtype)
            logits.real = logits.real - 0.5 * pt.logsumexp(2 * logits.real, dim=-1, keepdim=True)
            logits = pt.complex(
                pt.nan_to_num(logits.real, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype)),
                logits.imag)

        return logits

    @pt.no_grad()
    def sample(self, sample_num: int = None):
        x = pt.zeros((sample_num, self.qubit_num),
                     dtype=self.rdtype,
                     device=self.device)
        for qubit_idx in range(self.qubit_num):
            inp_x = 1 - 2 * x[..., :qubit_idx]
            logits = self.cond_log_abs(inp_x, qubit_idx)

            # Masking happens here:
            logits = self.mask_logits(logits, x[:, :qubit_idx].to(pt.long))
            probs = pt.exp(2 * logits)
            x[:, qubit_idx] = pt.bernoulli(probs[:, 1])
            if self.masking_mode == MaskingMode.part_base_vecs:
                mask = self.masker.mask(x[:, qubit_idx])
                x = x[mask]

        return self.base_vec2base_idx(x)

    @pt.no_grad()
    def sample_stats(self,
                     sample_num: int) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        unq_base_vecs = pt.zeros((1, 0),
                                 dtype=self.inp_dtype,
                                 device=self.device)
        unq_counts = pt.tensor([sample_num],
                               dtype=self.idx_dtype,
                               device=self.device)
        unq_weights = pt.tensor([1.0],
                                dtype=self.rdtype,
                                device=self.device)
        unq_acc_eigs = []
        if isinstance(self.masker, LocallyDecomposableMasker):
            for sym_idx in range(self.masker.sym_num):
                unq_acc_eigs.append(pt.zeros(unq_base_vecs.shape[:-1],
                                             dtype=self.idx_dtype,
                                             device=self.device) + self.masker.symmetries[sym_idx].start_eig)
            unq_acc_eigs = pt.stack(unq_acc_eigs, dim=-1)
        for qubit_idx in range(self.qubit_num):
            if unq_base_vecs.shape[0] > self.split_size:
                split_unq_base_vecs = pt.split(unq_base_vecs,
                                               split_size_or_sections=self.split_size,
                                               dim=0)
                split_unq_counts = pt.split(unq_counts,
                                            split_size_or_sections=self.split_size,
                                            dim=0)
                split_unq_weights = pt.split(unq_weights,
                                             split_size_or_sections=self.split_size,
                                             dim=0)
                if isinstance(self.masker, LocallyDecomposableMasker):
                    split_unq_acc_eigs = pt.split(unq_acc_eigs,
                                                  split_size_or_sections=self.split_size,
                                                  dim=0)

                unq_base_vecs, unq_counts, unq_weights, unq_acc_eigs = list(), list(), list(), list()
                for split_idx in range(len(split_unq_base_vecs)):
                    new_unq_base_vecs, new_unq_counts, new_unq_weights, new_unq_acc_eigs = self.sample_next_qubit_stats(qubit_idx=qubit_idx,
                                                                                                                        unq_base_vecs=split_unq_base_vecs[split_idx],
                                                                                                                        unq_counts=split_unq_counts[split_idx],
                                                                                                                        unq_weights=split_unq_weights[split_idx],
                                                                                                                        unq_acc_eigs=split_unq_acc_eigs[split_idx],
                                                                                                                        rng=self.rng)
                    unq_base_vecs.append(new_unq_base_vecs)
                    unq_counts.append(new_unq_counts)
                    unq_weights.append(new_unq_weights)
                    unq_acc_eigs.append(new_unq_acc_eigs)

                unq_base_vecs = pt.cat(unq_base_vecs, dim=0)
                unq_counts = pt.cat(unq_counts, dim=0)
                unq_weights = pt.cat(unq_weights, dim=0)
                if isinstance(self.masker, LocallyDecomposableMasker):
                    unq_acc_eigs = pt.cat(unq_acc_eigs, dim=0)
            else:
                unq_base_vecs, unq_counts, unq_weights, unq_acc_eigs = self.sample_next_qubit_stats(qubit_idx=qubit_idx,
                                                                                                    unq_base_vecs=unq_base_vecs,
                                                                                                    unq_counts=unq_counts,
                                                                                                    unq_weights=unq_weights,
                                                                                                    unq_acc_eigs=unq_acc_eigs,
                                                                                                    rng=self.rng)

        unq_indices = self.base_vec2base_idx(unq_base_vecs)
        if ((self.masking_mode == MaskingMode.logits) and (self.masking_depth == 0)):
            assert unq_counts.sum() == sample_num
        unq_counts = unq_counts.type(self.cdtype).to(self.device)
        unq_weights = unq_weights.type(self.cdtype).to(self.device)

        return unq_indices, unq_counts, unq_weights

    def sample_next_qubit_stats(self,
                                qubit_idx: int = None,
                                unq_base_vecs: pt.Tensor = None,
                                unq_counts: pt.Tensor = None,
                                unq_weights: pt.Tensor = None,
                                unq_acc_eigs: pt.Tensor = None,
                                rng: np.random.Generator = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        inp_x = 1 - 2 * unq_base_vecs
        logits = self.cond_log_abs(inp_x, qubit_idx)

        # Binomial sampling
        start_sample_num = unq_counts.sum()
        sampled_counts = rng.binomial(unq_counts.detach().cpu().numpy(),
                                      pt.exp(2 * logits).detach().cpu().numpy()[..., 1])
        sampled_counts = pt.tensor(sampled_counts,
                                   dtype=self.idx_dtype,
                                   device=self.device)
        sampled_counts = pt.stack((unq_counts - sampled_counts, sampled_counts), dim=-1)

        shape = unq_base_vecs.shape
        zero_outcome = pt.cat((unq_base_vecs,
                               pt.zeros((*shape[:-1], 1),
                                        dtype=self.rdtype,
                                        device=self.device)),
                              dim=-1)
        one_outcome = pt.cat((unq_base_vecs,
                              pt.ones((*shape[:-1], 1),
                                      dtype=self.rdtype,
                                      device=self.device)),
                             dim=-1)
        zero_outcome = pt.unsqueeze(zero_outcome, dim=1)
        one_outcome = pt.unsqueeze(one_outcome, dim=1)

        unq_base_vecs = pt.cat((zero_outcome, one_outcome), dim=1)
        unq_counts = sampled_counts
        unq_weights = pt.tile(pt.unsqueeze(unq_weights, dim=-1), dims=(1, 2))

        if isinstance(self.masker, LocallyDecomposableMasker):
            zero_outcome_eigs, zero_outcome_mask, one_outcome_eigs, one_outcome_mask = self.masker.acc_eigs2outcomes_masks(
                qubits_seen=qubit_idx,
                acc_eigs=unq_acc_eigs)
            unq_acc_eigs = pt.cat((pt.unsqueeze(zero_outcome_eigs, dim=-2),
                                   pt.unsqueeze(one_outcome_eigs, dim=-2)),
                                  dim=-2)
            phys_mask = pt.stack((zero_outcome_mask, one_outcome_mask), dim=-1)
        else:
            phys_mask = pt.ones(unq_base_vecs.shape[:-1], dtype=pt.bool, device=unq_base_vecs.device)

        if self.perseverant_sampling is True or ((self.masking_mode == MaskingMode.logits) and (qubit_idx < self.qubit_num - self.masking_depth)):
            co_sampled_counts = pt.flip(sampled_counts, dims=(-1,))
            co_mask = pt.logical_xor(phys_mask, pt.all(phys_mask, dim=-1, keepdim=True))
            unq_counts = pt.mul(sampled_counts, phys_mask) + pt.mul(co_sampled_counts, co_mask)
            if (self.perseverant_sampling is True) and (self.masking_mode != MaskingMode.logits):
                unq_weights = pt.mul(unq_weights,
                                     pt.div(sampled_counts.to(self.rdtype),
                                            unq_counts.to(self.rdtype)))
                unq_weights = pt.nan_to_num(unq_weights, nan=0.0)

        unq_base_vecs = unq_base_vecs[phys_mask]
        unq_counts = unq_counts[phys_mask]
        unq_weights = unq_weights[phys_mask]
        if isinstance(self.masker, LocallyDecomposableMasker):
            unq_acc_eigs = unq_acc_eigs[phys_mask]

        non_zero_count_mask = pt.logical_and(unq_counts > 0,
                                             ~pt.isclose(unq_weights,
                                                         pt.zeros_like(unq_weights)))
        unq_base_vecs = unq_base_vecs[non_zero_count_mask]
        unq_counts = unq_counts[non_zero_count_mask]
        unq_weights = unq_weights[non_zero_count_mask]
        if isinstance(self.masker, LocallyDecomposableMasker):
            unq_acc_eigs = unq_acc_eigs[non_zero_count_mask]

        if self.perseverant_sampling == 'renorm':
            unq_counts = pt.floor(unq_counts * (start_sample_num / unq_counts.sum())).type(self.idx_dtype)
        return unq_base_vecs, unq_counts, unq_weights, unq_acc_eigs
