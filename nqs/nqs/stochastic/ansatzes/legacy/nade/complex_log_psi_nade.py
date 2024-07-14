import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_COMPLEX_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.nade import AbstractNADE
from nqs.stochastic.ansatzes.legacy.anqs_primitives.nade import ComplexLogPsiMLP


class ComplexLogPsiNADE(AbstractNADE):
    def __init__(self,
                 *args,
                 depth: int = 0,
                 width: int = None,
                 activations: Union[Tuple[Callable], None] = None,
                 **kwargs):
        super(ComplexLogPsiNADE, self).__init__(*args, **kwargs)
        self.depth = depth
        self.width = width

        self.activations = None
        if activations is not None:
            assert len(activations) == self.depth + 1
            self.activations = activations

        self.log_psi_mlps = nn.ModuleList([ComplexLogPsiMLP(in_num=qubit_idx,
                                                            depth=self.depth,
                                                            width=self.width,
                                                            activations=self.activations,
                                                            dtype=self.cdtype)
                                           for qubit_idx in range(self.qubit_num)])

    @property
    def inp_dtype(self):
        return BASE_COMPLEX_TYPE

    def cond_log_psi(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:

        return self.log_psi_mlps[idx](x)

    def phase(self, x):
        raise DeprecationWarning(f'{self.__class__.__name__} object should never invoke phase '
                                 f'function')
