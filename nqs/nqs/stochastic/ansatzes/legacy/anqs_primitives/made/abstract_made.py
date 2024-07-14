import torch as pt

from abc import abstractmethod

from nqs.stochastic.ansatzes.legacy.anqs_primitives.abstract_anqs import AbstractANQS


class AbstractMADE(AbstractANQS):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AbstractMADE, self).__init__(*args, **kwargs)

    def log_psi(self, x: pt.Tensor):
        assert x.dtype == self.inp_dtype
        x_as_idx = x.type(pt.long)
        cond_log_psis = self.cond_log_psis(1 - 2 * x)

        return pt.sum(pt.squeeze(pt.gather(cond_log_psis, -1, pt.unsqueeze(x_as_idx, dim=-1)), dim=-1), dim=-1)

    @abstractmethod
    def cond_log_psis(self,
                      x: pt.Tensor = None) -> pt.Tensor:
        ...

    def cond_log_psi(self,
                     x: pt.Tensor = None,
                     idx: int = None) -> pt.Tensor:
        return self.cond_log_psis(x=x)[..., idx, :]
