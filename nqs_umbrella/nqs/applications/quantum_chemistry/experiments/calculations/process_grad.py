import numpy as np
import torch as pt

from .....infrastructure.nested_data import Config

from .sr import SRConfig, SRMetrics, sr
from .....stochastic.ansatzes.anqs import AbstractANQS
from .sample import SamplingResult


class ProcessGradConfig(Config):
    FIELDS = (
        'use_sr',
        'sr_config',
        'clip_grad_norm',
        'clip_grad_norm_value',
        'renorm_grad',
    )

    def __init__(self,
                 *args,
                 use_sr: bool = True,
                 sr_config: SRConfig = None,
                 clip_grad_norm: bool = True,
                 clip_grad_norm_value: float = 1.0,
                 renorm_grad: bool = False,
                 **kwargs):
        self.use_sr = use_sr
        self.sr_config = sr_config if sr_config is not None else SRConfig()
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_value = clip_grad_norm_value
        self.renorm_grad = renorm_grad

        super().__init__(*args, **kwargs)


class ProcessGradMetrics(Config):
    FIELDS = (
        'sr_metrics',
        'proc_grad_time',
    )

    def __init__(self,
                 *args,
                 sr_metrics: SRMetrics = None,
                 proc_grad_time: float = np.nan,
                 **kwargs):
        self.sr_metrics = sr_metrics if sr_metrics is not None else SRMetrics()

        self.proc_grad_time = proc_grad_time

        super().__init__(*args, **kwargs)


def process_grad(wf: AbstractANQS = None,
                 sampling_result: SamplingResult = None,
                 config: ProcessGradConfig = None):
    proc_grad_metrics = ProcessGradMetrics()
    if config.use_sr:
        wf.cat_grad, proc_grad_metrics.sr_metrics, sr_time = sr(wf=wf,
                                                                sampling_result=sampling_result,
                                                                config=config.sr_config)
        proc_grad_metrics.sr_metrics.sr_time = sr_time
    if config.clip_grad_norm:
        wf.clip_grad_norm(config.clip_grad_norm_value)

    if config.renorm_grad:
        wf.cat_grad = wf.cat_grad / pt.linalg.norm(wf.cat_grad)

    return proc_grad_metrics
