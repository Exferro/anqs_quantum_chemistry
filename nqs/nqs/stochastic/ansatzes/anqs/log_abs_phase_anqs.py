import torch as pt
from torch import nn

from typing import Union

from ....base.constants import NEGINF
from ....base.constants import BASE_REAL_TYPE

from .abstract_anqs import ANQSConfig, AbstractANQS

from .mlp import MLP


class LogAbsPhaseANQS(AbstractANQS):
    def __init__(self,
                 *args,
                 config: ANQSConfig = None,
                 **kwargs):
        super(LogAbsPhaseANQS, self).__init__(*args,
                                              config=config,
                                              **kwargs)
        assert self.dtype == BASE_REAL_TYPE

        if self.de_mode == 'NADE':
            self.log_abs_subnet = nn.ModuleList(
                [MLP(in_num=self.qudit_ends[qudit_idx - 1] if qudit_idx != 0 else 1,
                     is_made=False,
                     qubit_grouping=self.qubit_grouping,
                     out_num=self.qudit_dims[qudit_idx],
                     dtype=self.dtype,
                     is_out_complex=False,
                     config=self.config.main_subnet_config)
                 for qudit_idx in range(self.qudit_num)])
            self.phase_subnet = nn.ModuleList(
                [MLP(in_num=self.qudit_ends[qudit_idx - 1] if qudit_idx != 0 else 1,
                     is_made=False,
                     qubit_grouping=self.qubit_grouping,
                     out_num=self.qudit_dims[qudit_idx],
                     dtype=self.dtype,
                     is_out_complex=False,
                     config=self.config.aux_subnet_config)
                 for qudit_idx in range(self.qudit_num)])

        elif self.de_mode == 'MADE':
            self.log_abs_subnet = MLP(in_num=self.qubit_num,
                                      is_made=True,
                                      qubit_grouping=self.qubit_grouping,
                                      dtype=self.dtype,
                                      is_out_complex=False,
                                      config=self.config.main_subnet_config)
            self.phase_subnet = MLP(in_num=self.qubit_num,
                                    is_made=True,
                                    qubit_grouping=self.qubit_grouping,
                                    dtype=self.dtype,
                                    is_out_complex=False,
                                    config=self.config.aux_subnet_config)

    def clip_grad_norm(self, value: float = None):
        #pt.nn.utils.clip_grad_norm_(self.log_abs_subnet.parameters(), value)
        #pt.nn.utils.clip_grad_norm_(self.phase_subnet.parameters(), value)
        pt.nn.utils.clip_grad_norm_(self.parameters(), value)

    def _cond_log_abs(self,
                      qudit_idx: Union[int, None] = None,
                      base_vec: pt.Tensor = None) -> pt.Tensor:
        if self.de_mode == 'NADE':
            x = self.log_abs_subnet[qudit_idx].align_input(x=base_vec, x_is_base_vec=True)
            cond_log_abs = self.log_abs_subnet[qudit_idx](x)

        elif self.de_mode == 'MADE':
            assert qudit_idx is None
            x = self.log_abs_subnet.align_input(x=base_vec, x_is_base_vec=True)
            cond_log_abs = self.log_abs_subnet(x)
        else:
            raise RuntimeError(f'Somehow we have wrong de_mode: {self.de_mode}')

        return cond_log_abs

    def _cond_phase(self,
                    qudit_idx: Union[int, None] = None,
                    base_vec: pt.Tensor = None) -> pt.Tensor:
        if self.de_mode == 'NADE':
            x = self.phase_subnet[qudit_idx].align_input(x=base_vec, x_is_base_vec=True)
            cond_phase = self.phase_subnet[qudit_idx](x)
            # if self.spin_flip_symmetry:
            #     sf_base_vec = self.spin_flip_base_vec(pt.clone(base_vec))
            #     sf_x = self.phase_subnet[qudit_idx].align_input(x=sf_base_vec, x_is_base_vec=True)
            #     sf_cond_phase = self.phase_subnet[qudit_idx](sf_x)
            #     cond_phase = 0.5 * (cond_phase + sf_cond_phase[..., self.sf_qudit_couplings[qudit_idx]])
        elif self.de_mode == 'MADE':
            assert qudit_idx is None
            x = self.phase_subnet.align_input(x=base_vec, x_is_base_vec=True)
            cond_phase = self.phase_subnet(x)
        else:
            raise RuntimeError(f'Somehow we have wrong de_mode: {self.de_mode}')

        return pt.tensor(pt.pi, dtype=self.rdtype, device=self.device) * cond_phase

    def _cond_log_psi(self,
                      qudit_idx: Union[int, None] = None,
                      base_vec: pt.Tensor = None) -> pt.Tensor:
        return pt.complex(self._cond_log_abs(qudit_idx=qudit_idx, base_vec=base_vec),
                          self._cond_phase(qudit_idx=qudit_idx, base_vec=base_vec))

    def cond_log_abs(self,
                     qudit_idx: int = None,
                     base_vec: pt.Tensor = None,
                     return_all_if_made: bool = False,
                     mask: pt.Tensor = None) -> pt.Tensor:
        assert qudit_idx is not None
        if self.de_mode == 'NADE':
            cond_log_abs = self._cond_log_abs(qudit_idx=qudit_idx, base_vec=base_vec)
            #print(f'I am subtracting mean from the last log abses!')
            if self.config.subtract_mean:
                cond_log_abs = cond_log_abs - pt.mean(cond_log_abs, dim=-1, keepdim=True)
            if self.spin_flip_symmetry_config.abs:
                sf_base_vec = self.spin_flip_base_vec(pt.clone(base_vec))
                sf_cond_log_abs = self._cond_log_abs(qudit_idx=qudit_idx, base_vec=sf_base_vec)
                if self.config.subtract_mean:
                    sf_cond_log_abs = sf_cond_log_abs - pt.mean(sf_cond_log_abs, dim=-1, keepdim=True)
                cond_log_abs = 0.5 * (cond_log_abs + sf_cond_log_abs[..., self.sf_qudit_couplings[qudit_idx]])

            assert mask is not None
            cond_log_abs = pt.where(mask,
                                    cond_log_abs,
                                    pt.full(cond_log_abs.shape,
                                            NEGINF,
                                            dtype=cond_log_abs.dtype,
                                            device=cond_log_abs.device))

            cond_log_abs = self.normalise_cond_log(cond_log_abs)

        elif self.de_mode == 'MADE':
            cond_log_abs = self._cond_log_abs(qudit_idx=None, base_vec=base_vec)
            if self.config.subtract_mean:
                cond_log_abs = cond_log_abs - pt.mean(cond_log_abs, dim=-1, keepdim=True)
            # if self.spin_flip_symmetry_config.abs:
            #     sf_base_vec = self.spin_flip_base_vec(pt.clone(base_vec))
            #     sf_cond_log_abs = self._cond_log_abs(qudit_idx=None, base_vec=sf_base_vec)
            #     if self.anqs_config.subtract_mean:
            #         sf_cond_log_abs = sf_cond_log_abs - pt.mean(sf_cond_log_abs, dim=-1, keepdim=True)
            #     cond_log_abs = 0.5 * (cond_log_abs + sf_cond_log_abs[..., self.sf_qudit_couplings[0]])
            
            # cond_log_abs = pt.where(self.made_qudit_dim_mask,
            #                         cond_log_abs,
            #                         pt.full(cond_log_abs.shape, NEGINF, dtype=cond_log_abs.dtype,
            #                                 device=cond_log_abs.device))
            if return_all_if_made:
                cond_log_abs = cond_log_abs[..., :qudit_idx + 1, :]
            else:
                cond_log_abs = cond_log_abs[..., qudit_idx, :]

            assert mask is not None
            cond_log_abs = pt.where(mask,
                                    cond_log_abs,
                                    pt.full(cond_log_abs.shape, NEGINF, dtype=cond_log_abs.dtype,
                                            device=cond_log_abs.device))

            cond_log_abs = self.normalise_cond_log(cond_log_abs)
        else:
            raise RuntimeError(f'Somehow we have wrong de_mode: {self.de_mode}')

        return cond_log_abs
