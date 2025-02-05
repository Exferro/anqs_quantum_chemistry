import torch as pt

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_COMPLEX_TYPE

from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.abstract_made import AbstractMADE
from nqs.stochastic.ansatzes.legacy.anqs_primitives.made.complex_log_psi_made_mlp import ComplexLogPsiMADEMLP


class ComplexLogPsiMADE(AbstractMADE):
    def __init__(self,
                 *args,
                 depth: int = 0,
                 width: int = None,
                 activations: Union[Tuple[Callable], None] = None,
                 **kwargs):
        super(ComplexLogPsiMADE, self).__init__(*args, **kwargs)
        self.depth = depth
        self.width = width

        self.activations = None
        if activations is not None:
            assert len(activations) == self.depth + 1
            self.activations = activations

        self.log_psi_mlp = ComplexLogPsiMADEMLP(in_num=self.qubit_num,
                                                depth=self.depth,
                                                width=self.width,
                                                activations=self.activations,
                                                dtype=self.cdtype)

    @property
    def inp_dtype(self):
        return BASE_COMPLEX_TYPE

    def cond_log_psis(self,
                      x: pt.Tensor = None) -> pt.Tensor:

        return self.log_psi_mlp(x)
