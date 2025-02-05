import torch as pt
import numpy as np

from .....infrastructure.nested_data import Config
from .....infrastructure.timed_decorator import timed

from .....stochastic.ansatzes.anqs import AbstractANQS

from .sample import SamplingResult


class SRConfig(Config):
    FIELDS = (
        'max_indices_num',
        'use_theor_freqs',
        'use_reg',
        'reg_eps'
    )

    def __init__(self,
                 *args,
                 max_indices_num: int = 25,
                 use_theor_freqs: bool = False,
                 use_reg: bool = True,
                 reg_eps: float = 1e-4,
                 **kwargs):
        self.max_indices_num = max_indices_num
        self.use_theor_freqs = use_theor_freqs
        self.use_reg = use_reg
        self.reg_eps = reg_eps

        super().__init__(*args, **kwargs)


class SRMetrics(Config):
    FIELDS = (
        'sr_unq_num',
        'sr_sampled_prob',
        'sr_max_amp',
        'sr_min_amp',
        'sr_time'
    )

    def __init__(self,
                 *args,
                 sr_unq_num: float = np.nan,
                 sr_sampled_prob: float = np.nan,
                 sr_max_amp: float = np.nan,
                 sr_min_amp: float = np.nan,
                 sr_time: float = np.nan,
                 **kwargs):
        self.sr_unq_num = sr_unq_num
        self.sr_sampled_prob = sr_sampled_prob
        self.sr_max_amp = sr_max_amp
        self.sr_min_amp = sr_min_amp

        self.sr_time = sr_time

        super().__init__(*args, **kwargs)


def hermitian_matmul(a):
    outer_size = a.shape[0]
    result = pt.zeros((outer_size, outer_size), dtype=a.dtype, device=a.device)
    for idx in range(outer_size):
        result[idx, idx:] = a[idx:idx + 1, :] @ a[idx:, :].T.conj()

    return result + pt.tril(result.T.conj(), diagonal=-1)


def soft_eigvals_inv(eigvals):
    eigvals_inv = 1 / eigvals
    return pt.where(pt.isclose(eigvals, pt.zeros_like(eigvals)),
                    pt.zeros_like(eigvals),
                    eigvals_inv)


def soft_matrix_inv(a):
    u, s, v = np.linalg.svd(a.cpu().numpy(), full_matrices=True)
    u = pt.from_numpy(u).to(a.device)
    s = pt.from_numpy(s).to(a.device)
    v = pt.from_numpy(v).to(a.device)
    s_inv = soft_eigvals_inv(s)

    return v.T.conj() @ pt.diag(s_inv.type(a.dtype)) @ u.T.conj()


@timed
@pt.no_grad()
def sr(wf: AbstractANQS = None,
       sampling_result: SamplingResult = None,
       config: SRConfig = None):
    config = config if config is not None else SRConfig()
    sr_metrics = SRMetrics()

    cat_grad = wf.cat_grad
    cat_grad = pt.complex(cat_grad,
                          pt.zeros_like(cat_grad))

    if config.use_theor_freqs:
        amps = wf.amplitude(sampling_result.indices)
        freqs = pt.conj(amps) * amps
    else:
        freqs = sampling_result.counts
    freqs = freqs / pt.sum(freqs)

    sorted_freqs, sorted_freq_inv = pt.sort(freqs.real, descending=True)
    sr_freqs = freqs[sorted_freq_inv[:config.max_indices_num]]

    sr_freqs = sr_freqs / pt.sum(sr_freqs)
    sr_indices = sampling_result.indices[sorted_freq_inv[:config.max_indices_num]]
    sr_amps = wf.amplitude(sr_indices)
    sr_metrics.sr_unq_num = sr_amps.shape[0]
    sr_metrics.sr_sampled_prob = pt.dot(pt.conj(sr_amps), sr_amps).cpu().real.item()
    sr_metrics.sr_max_amp = sr_amps[0].cpu().item()
    sr_metrics.sr_min_amp = sr_amps[sr_metrics.sr_unq_num - 1].cpu().item()

    cat_log_jac = wf.compute_cat_log_jac(sr_indices)
    cat_log_jac = cat_log_jac - (cat_log_jac.T * sr_freqs).sum(dim=-1, keepdim=True).T

    if config.use_reg:
        reg_eps_inv = 1 / config.reg_eps
        o_matrix = reg_eps_inv * pt.diag(pt.sqrt(sr_freqs)) @ cat_log_jac.conj()

        t_matrix = hermitian_matmul(o_matrix)
        t_matrix_reg = pt.diag(pt.ones_like(pt.diag(t_matrix))) + config.reg_eps * t_matrix
        cat_grad = reg_eps_inv * cat_grad - (o_matrix.T.conj() @  pt.linalg.solve(t_matrix_reg, o_matrix @ cat_grad))
    else:
        o_matrix = pt.diag(pt.sqrt(sr_freqs)) @ cat_log_jac.conj()

        t_matrix = hermitian_matmul(o_matrix)
        t_matrix_inv = soft_matrix_inv(t_matrix)

        cat_grad = (o_matrix.T.conj() @ (t_matrix_inv.T.conj() @ (t_matrix_inv @ (o_matrix @ cat_grad))))

    return cat_grad.real, sr_metrics
