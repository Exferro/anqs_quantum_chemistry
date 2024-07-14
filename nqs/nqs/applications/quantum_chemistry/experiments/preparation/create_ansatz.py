import torch as pt

from .....base.hilbert_space import HilbertSpace
from .....infrastructure.nested_data import Config

from .....stochastic.ansatzes.anqs import ANQSConfig
from .....stochastic.ansatzes.anqs import LogPsiANQS
from .....stochastic.ansatzes.anqs import LogAbsPhaseANQS

from .....stochastic.maskers import LocallyDecomposableMasker


class MetaAnsatzConfig(Config):
    ALLOWED_TYPES = ('LogPsiANQS', 'LogAbsPhaseANQS')
    FIELDS = (
        'ansatz_type',
        'ansatz_config',
    )

    def __init__(self,
                 *args,
                 ansatz_type: str = 'LogAbsPhaseANQS',
                 ansatz_config: ANQSConfig = None,
                 **kwargs):
        assert ansatz_type in self.ALLOWED_TYPES
        self._ansatz_type = ansatz_type
        self.ansatz_config = ansatz_config if ansatz_config is not None else ANQSConfig()

        super().__init__(*args, **kwargs)

    @property
    def ansatz_type(self):
        return self._ansatz_type

    @ansatz_type.setter
    def ansatz_type(self, value: str = None):
        assert value in self.ALLOWED_TYPES
        self._ansatz_type = value


def create_ansatz(config: MetaAnsatzConfig = None,
                  hs: HilbertSpace = None,
                  masker: LocallyDecomposableMasker = None,
                  sign_structure: pt.Tensor = None):
    config = config if config is not None else MetaAnsatzConfig()
    if config.ansatz_type == 'LogPsiANQS':
        return LogPsiANQS(hilbert_space=hs,
                          masker=masker,
                          sign_structure=sign_structure,
                          config=config.ansatz_config)
    elif config.ansatz_type == 'LogAbsPhaseANQS':
        return LogAbsPhaseANQS(hilbert_space=hs,
                               masker=masker,
                               sign_structure=sign_structure,
                               config=config.ansatz_config)
    else:
        raise RuntimeError(f'Wrong ansatz type: {config.ansatz_type}')
