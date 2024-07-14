import torch as pt
import numpy as np

from typing import Tuple

from nqs.base.constants import BASE_REAL_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.abstract_anqs import MaskingMode

from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.abstract_made import AbstractMADE
from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.transformer_made import TransformerMADE


class RealLogPsiTransformerMADE(AbstractMADE):
    def __init__(self,
                 *args,
                 dim: int = None,
                 depth: int = 0,
                 width: int = 512,
                 head_num: int = None,
                 **kwargs):
        super(RealLogPsiTransformerMADE, self).__init__(*args, **kwargs)
        self.dim = dim
        self.depth = depth
        self.head_num = head_num
        self.width = width

        self.transformer_made = TransformerMADE(dim=self.dim,
                                                out_dim=4,
                                                depth=self.depth,
                                                qubit_num=self.qubit_num,
                                                head_num=self.head_num,
                                                dtype=self.rdtype)

    @property
    def inp_dtype(self):
        return BASE_REAL_TYPE

    def cond_log_psis(self, x: pt.Tensor = None) -> pt.Tensor:
        x_as_idx = x.type(pt.long)
        cond_log_psis = self.transformer_made(x_as_idx)
        cond_log_psis = pt.reshape(cond_log_psis, (*cond_log_psis.shape[:-1], 2, 2))
        log_psis = pt.view_as_complex(cond_log_psis)

        cond_log_psis = log_psis - 0.5 * pt.complex(pt.logsumexp(2 * log_psis.real, dim=-1, keepdim=True),
                                                    pt.zeros_like(log_psis.imag))

        return cond_log_psis

    def log_psi(self, x: pt.Tensor):
        x_as_idx = x.type(pt.long)
        cond_log_psis = self.cond_log_psis(x_as_idx)[:, :x.shape[-1], :]

        return pt.sum(pt.squeeze(pt.gather(cond_log_psis, -1, pt.unsqueeze(x_as_idx, dim=-1)), dim=-1), dim=-1)

    def phase(self, x):
        raise DeprecationWarning(f'{self.__class__.__name__} object should never invoke phase '
                                 f'function')

    def sample_next_qubit_stats(self,
                                qubit_idx: int = None,
                                unq_base_vecs: pt.Tensor = None,
                                unq_counts: pt.Tensor = None,
                                unq_weights: pt.Tensor = None,
                                unq_acc_eigs: pt.Tensor = None,
                                rng: np.random.Generator = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        logits = self.cond_log_abs(unq_base_vecs, qubit_idx)

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

        if self.masker is not None:
            zero_outcome_eigs, zero_outcome_mask, one_outcome_eigs, one_outcome_mask = self.masker.acc_eigs2outcomes_masks(
                qubits_seen=qubit_idx,
                acc_eigs=unq_acc_eigs)
            unq_acc_eigs = pt.cat((pt.unsqueeze(zero_outcome_eigs, dim=-2),
                                   pt.unsqueeze(one_outcome_eigs, dim=-2)),
                                  dim=-2)
            phys_mask = pt.stack((zero_outcome_mask, one_outcome_mask), dim=-1)
            #phys_mask = self.masker.mask(unq_base_vecs.type(self.idx_dtype))
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
        if self.masker is not None:
            unq_acc_eigs = unq_acc_eigs[phys_mask]

        non_zero_count_mask = pt.logical_and(unq_counts > 0,
                                             ~pt.isclose(unq_weights,
                                                         pt.zeros_like(unq_weights)))
        unq_base_vecs = unq_base_vecs[non_zero_count_mask]
        unq_counts = unq_counts[non_zero_count_mask]
        unq_weights = unq_weights[non_zero_count_mask]
        if self.masker is not None:
            unq_acc_eigs = unq_acc_eigs[non_zero_count_mask]

        if self.perseverant_sampling == 'renorm':
            unq_counts = pt.floor(unq_counts * (start_sample_num / unq_counts.sum())).type(self.idx_dtype)
        return unq_base_vecs, unq_counts, unq_weights, unq_acc_eigs
