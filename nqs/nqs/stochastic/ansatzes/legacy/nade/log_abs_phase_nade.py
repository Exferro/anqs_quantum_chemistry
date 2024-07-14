import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_REAL_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.nade import AbstractNADE
from nqs.stochastic.ansatzes.legacy.anqs_primitives.mlp import MLP
from nqs.stochastic.ansatzes.legacy.anqs_primitives.nade.log_abs_mlp import LogAbsMLP


class LogAbsPhaseNADE(AbstractNADE):
    def __init__(self,
                 *args,
                 depth: int = 0,
                 width: int = None,
                 log_abs_activations: Union[Tuple[Callable], None] = None,
                 phase_depth: int = 2,
                 phase_width: int = 512,
                 phase_activations: Union[Tuple[Callable], None] = None,
                 **kwargs):
        super(LogAbsPhaseNADE, self).__init__(*args, **kwargs)

        self.depth = depth
        self.width = width

        self.log_abs_activations = None
        if log_abs_activations is not None:
            assert len(log_abs_activations) == self.depth + 1
            self.log_abs_activations = log_abs_activations

        self.log_abs_mlps = nn.ModuleList([LogAbsMLP(in_num=qubit_idx,
                                                     depth=self.depth,
                                                     width=self.width,
                                                     activations=self.log_abs_activations,
                                                     dtype=self.rdtype)
                                           for qubit_idx in range(self.qubit_num)])

        self.phase_depth = phase_depth
        self.phase_width = phase_width

        self.phase_activations = None
        if phase_activations is not None:
            assert len(phase_activations) == self.phase_depth + 1
            self.phase_activations = phase_activations

        self.phase_mlps = nn.ModuleList([MLP(in_num=qubit_idx,
                                             out_num=2,
                                             depth=self.phase_depth,
                                             width=self.phase_width,
                                             activations=self.phase_activations,
                                             dtype=self.rdtype)
                                         for qubit_idx in range(self.qubit_num)])

    @property
    def inp_dtype(self):
        return BASE_REAL_TYPE

    def amplitude(self, base_idx: pt.Tensor) -> pt.Tensor:
        base_vec = self.base_idx2base_vec(base_idx).type(self.inp_dtype).to(self.device)
        return pt.exp(self.log_abs(base_vec).type(self.cdtype)
                      + 1j * self.phase(base_vec).type(self.cdtype))

    def cond_log_psi(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:

        raise DeprecationWarning(f'{self.__class__.__name__} object should never invoke cond_log_psi '
                                 f'function')

    def log_abs(self, x: pt.Tensor):
        assert x.dtype == self.inp_dtype
        x_as_idx = x.type(pt.long)
        log_abses = pt.zeros(*x.shape[:-1],
                             dtype=self.rdtype,
                             device=self.device)
        for qubit_idx in range(self.qubit_num):
            inp_x = 1 - 2 * x[..., :qubit_idx]
            logits = self.cond_log_abs(inp_x, qubit_idx)

            # Masking happens here:
            logits = self.mask_logits(logits, x_as_idx[:, :qubit_idx])
            log_abses = log_abses + pt.squeeze(pt.gather(logits,
                                                         dim=1,
                                                         index=x_as_idx[:, qubit_idx:qubit_idx + 1]))

        return log_abses

    def cond_log_abs(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:

        return self.log_abs_mlps[idx].log_abs(x)

    def phase(self, x):
        assert x.dtype == self.inp_dtype
        x_as_idx = x.type(pt.long)
        phases = pt.zeros(*x.shape[:-1],
                          dtype=self.rdtype,
                          device=self.device)
        for qubit_idx in range(self.qubit_num):
            inp_x = 1 - 2 * x[..., :qubit_idx]
            cond_phases = self.cond_phase(inp_x, qubit_idx)

            phases = phases + pt.squeeze(pt.gather(cond_phases,
                                                   dim=1,
                                                   index=x_as_idx[:, qubit_idx:qubit_idx + 1]))

        return phases

    def cond_phase(self,
                   x: pt.Tensor = None,
                   idx: int = None) -> pt.Tensor:

        return self.phase_mlps[idx](x)

