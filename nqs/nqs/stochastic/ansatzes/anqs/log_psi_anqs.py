import torch as pt
from torch import nn

from typing import Union

from ....base.constants import BASE_REAL_TYPE

from .abstract_anqs import ANQSConfig, AbstractANQS

from .mlp import MLP


class LogPsiANQS(AbstractANQS):
    def __init__(self,
                 *args,
                 config: ANQSConfig = None,
                 **kwargs):
        super(LogPsiANQS, self).__init__(*args,
                                         config=config,
                                         **kwargs)
        if self.de_mode == 'NADE':
            self.log_psi_subnet = nn.ModuleList(
                [MLP(in_num=self.qudit_ends[qudit_idx - 1] if qudit_idx != 0 else 1,
                     is_made=False,
                     qubit_grouping=self.qubit_grouping,
                     out_num=self.qudit_dims[qudit_idx],
                     dtype=self.dtype,
                     is_out_complex=True,
                     config=self.config.main_subnet_config)
                 for qudit_idx in range(self.qudit_num)])
        elif self.de_mode == 'MADE':
            self.log_psi_subnet = MLP(in_num=self.qubit_num,
                                      is_made=True,
                                      qubit_grouping=self.qubit_grouping,
                                      dtype=self.dtype,
                                      is_out_complex=True,
                                      config=self.config.main_subnet_config)

    def clip_grad_norm(self, value: float = None):
        pt.nn.utils.clip_grad_norm_(self.log_psi_subnet.parameters(), value)

    def _cond_log_psi(self,
                      qudit_idx: Union[int, None] = None,
                      base_vec: pt.Tensor = None) -> pt.Tensor:
        if self.de_mode == 'NADE':
            x = self.log_psi_subnet[qudit_idx].align_input(x=base_vec, x_is_base_vec=True)
            cond_log_psi = self.log_psi_subnet[qudit_idx](x)
        elif self.de_mode == 'MADE':
            assert qudit_idx is None
            x = self.log_psi_subnet.align_input(x=base_vec, x_is_base_vec=True)
            cond_log_psi = self.log_psi_subnet(x)
        else:
            raise RuntimeError(f'Somehow we have wrong de_mode: {self.de_mode}')

        if self.dtype == BASE_REAL_TYPE:
            cond_log_psi = pt.reshape(cond_log_psi, (*cond_log_psi.shape[:-1], -1, 2))
            cond_log_psi = pt.view_as_complex(cond_log_psi)

        return cond_log_psi
