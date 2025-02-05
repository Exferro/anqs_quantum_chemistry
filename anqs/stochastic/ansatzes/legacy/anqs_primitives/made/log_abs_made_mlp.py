import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_REAL_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.made_mlp import MADEMLP


class LogAbsMADEMLP(nn.Module):
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
                 dtype=BASE_REAL_TYPE):
        super(LogAbsMADEMLP, self).__init__()
        self.dtype = dtype

        self.log_abs_mlp = MADEMLP(in_num=in_num,
                                   out_num=2,
                                   depth=depth,
                                   width=width,
                                   activations=activations,
                                   dtype=self.dtype)

    def log_abs(self, x):
        log_abses = self.log_abs_mlp(x)
        
        # We normalize log-absolute values of complex amplitude by using
        # a numerically stable logsumexp function
        log_abses = log_abses - 0.5 * pt.logsumexp(2 * log_abses, dim=-1, keepdim=True)

        return log_abses

    def forward(self, x):
        return self.log_abs(x)
