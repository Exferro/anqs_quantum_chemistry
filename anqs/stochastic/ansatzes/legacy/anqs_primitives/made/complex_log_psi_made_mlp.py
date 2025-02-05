import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_COMPLEX_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.made_mlp import MADEMLP


class ComplexLogPsiMADEMLP(nn.Module):
    """
    A particular class of MLPs which output normalized log-amplitudes, where amplitude absolute
    value and phase are calculated by two separate subnetworks.
    A user doesn't have to care about any real to complex conversions,
    everything is done under the hood.
    """

    def __init__(self,
                 in_num: int = None,
                 depth: int = 0,
                 width: int = None,
                 activations: Union[Tuple[Callable], None] = None,
                 dtype=BASE_COMPLEX_TYPE):
        super(ComplexLogPsiMADEMLP, self).__init__()
        self.dtype = dtype

        self.log_psi_mlp = MADEMLP(in_num=in_num,
                                   out_num=2,
                                   depth=depth,
                                   width=width,
                                   activations=activations,
                                   dtype=self.dtype)

    def log_abs(self, x):

        return self(x).real

    def phase(self, x):

        return self(x).imag

    def forward(self, x):
        log_psis = self.log_psi_mlp(x)
        log_psis = log_psis - 0.5 * pt.complex(pt.logsumexp(2 * log_psis.real, dim=-1, keepdim=True),
                                               pt.zeros_like(log_psis.imag))

        return log_psis
