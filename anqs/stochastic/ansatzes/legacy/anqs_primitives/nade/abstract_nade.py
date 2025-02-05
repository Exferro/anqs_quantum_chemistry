import torch as pt

from nqs.stochastic.ansatzes.legacy.anqs_primitives.abstract_anqs import AbstractANQS, MaskingMode


class AbstractNADE(AbstractANQS):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AbstractNADE, self).__init__(*args, **kwargs)

    def log_psi(self, x: pt.Tensor):
        assert x.dtype == self.inp_dtype
        x_as_idx = x.type(pt.long)
        log_psis = pt.zeros(*x.shape[:-1],
                            dtype=self.cdtype,
                            device=self.device)
        if (self.masker is not None) and (self.masking_mode == MaskingMode.logits):
            part_outcome_masks = self.masker.compute_rolling_outcome_masks(x_as_idx)
        for qubit_idx in range(self.qubit_num):
            inp_x = 1 - 2 * x[..., :qubit_idx]
            logits = self.cond_log_psi(inp_x, qubit_idx)

            # Masking happens here:
            if (self.masker is not None) and (self.masking_mode == MaskingMode.logits) and (
                    qubit_idx < self.qubit_num - self.masking_depth):
                logits = self.mask_logits_given_mask(logits, part_outcome_masks[qubit_idx])
            log_psis = log_psis + pt.squeeze(pt.gather(logits,
                                                       dim=1,
                                                       index=x_as_idx[:, qubit_idx:qubit_idx + 1]))

        return log_psis
