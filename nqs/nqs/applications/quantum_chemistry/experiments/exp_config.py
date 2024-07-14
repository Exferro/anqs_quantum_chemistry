
from nqs.infrastructure.nested_data import Config

from ..molecule import MolConfig
from .preparation import MaskerConfig, MetaAnsatzConfig
from .calculations import SamplingConfig

from typing import Tuple


class ExpConfig(Config):
    FIELDS = (
        'mol_config',
        'perm_type',
        'masker_config',
        'meta_ansatz_config',
        'sampling_schedule',
        'rng_seed',
    )

    def __init__(self,
                 *args,
                 mol_config: MolConfig = None,
                 perm_type: str = 'direct',
                 masker_config: MaskerConfig = None,
                 meta_ansatz_config: MetaAnsatzConfig = None,
                 sampling_schedule: Tuple[int, SamplingConfig] = None,
                 rng_seed: int = 0,
                 **kwargs):
        self.mol_config = mol_config
        self.perm_type = perm_type
        self.masker_config = masker_config if masker_config is not None else MaskerConfig()
        self.meta_ansatz_config = meta_ansatz_config if meta_ansatz_config is not None else MetaAnsatzConfig()
        self.sampling_schedule = sampling_schedule if sampling_schedule is not None else ((0, SamplingConfig()),)
        self.rng_seed = rng_seed

        super().__init__(*args, **kwargs)
