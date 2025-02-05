import torch as pt

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_REAL_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.abstract_made import AbstractMADE
from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.made_mlp import MADEMLP
from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.log_abs_made_mlp import LogAbsMADEMLP


class LogAbsPhaseMADE(AbstractMADE):
    def __init__(self,
                 *args,
                 depth: int = 0,
                 width: int = None,
                 log_abs_activations: Union[Tuple[Callable], None] = None,
                 phase_depth: int = 2,
                 phase_width: int = 512,
                 phase_activations: Union[Tuple[Callable], None] = None,
                 **kwargs):
        super(LogAbsPhaseMADE, self).__init__(*args, **kwargs)

        self.depth = depth
        self.width = width

        self.log_abs_activations = None
        if log_abs_activations is not None:
            assert len(log_abs_activations) == self.depth + 1
            self.log_abs_activations = log_abs_activations

        self.log_abs_mlp = LogAbsMADEMLP(in_num=self.qubit_num,
                                         depth=self.depth,
                                         width=self.width,
                                         activations=self.log_abs_activations,
                                         dtype=self.rdtype)

        self.phase_depth = phase_depth
        self.phase_width = phase_width

        self.phase_activations = None
        if phase_activations is not None:
            assert len(phase_activations) == self.phase_depth + 1
            self.phase_activations = phase_activations

        self.phase_mlp = MADEMLP(in_num=self.qubit_num,
                                 out_num=2,
                                 depth=self.phase_depth,
                                 width=self.phase_width,
                                 activations=self.phase_activations,
                                 dtype=self.rdtype)

    @property
    def inp_dtype(self):
        return BASE_REAL_TYPE

    def amplitude(self, base_idx: pt.Tensor) -> pt.Tensor:
        base_vec = self.base_idx2base_vec(base_idx).type(self.inp_dtype).to(self.device)
        return pt.exp(self.log_abs(base_vec).type(self.cdtype)
                      + 1j * self.phase(base_vec).type(self.cdtype))

    def cond_log_psis(self,
                      x: pt.Tensor = None,
                      idx: int = None) -> pt.Tensor:

        raise DeprecationWarning(f'{self.__class__.__name__} object should never invoke cond_log_psi '
                                 f'function')

    def log_abs(self,
                x: pt.Tensor = None) -> pt.Tensor:
        assert x.dtype == self.inp_dtype
        x_as_idx = x.type(pt.long)
        cond_log_abses = self.cond_log_abses(1 - 2 * x)

        return pt.sum(pt.squeeze(pt.gather(cond_log_abses, -1, pt.unsqueeze(x_as_idx, dim=-1)), dim=-1), dim=-1)

    def cond_log_abses(self,
                       x: pt.Tensor = None) -> pt.Tensor:
        return self.log_abs_mlp(x)

    def cond_log_abs(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:

        return self.cond_log_abses(x=x)[..., idx, :]

    def phase(self, x):
        assert x.dtype == self.inp_dtype
        x_as_idx = x.type(pt.long)
        cond_phases = self.cond_phases(1 - 2 * x)

        return pt.sum(pt.squeeze(pt.gather(cond_phases, -1, pt.unsqueeze(x_as_idx, dim=-1)), dim=-1), dim=-1)

    def cond_phases(self,
                    x: pt.Tensor = None) -> pt.Tensor:
        return self.phase_mlp(x)

    def cond_phase(self,
                   x: pt.Tensor = None,
                   idx: int = None) -> pt.Tensor:

        return self.cond_phases(x=x)[..., idx, :]

