from ....infrastructure.nested_data import Config

LOCAL_SAMPLING_STRATEGIES = ('DU', 'MU',)


class LocalSamplingConfig(Config):
    FIELDS = (
        'type',
        'strategy',
        'depth',
    )
    DEFAULT_VALUES = {
        'type': 'uniform',
        'strategy': 'MU',
        'depth': '2',
    }
