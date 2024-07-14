import torch as pt

from .....infrastructure.nested_data import Config


class OptConfig(Config):
    ALLOWED_TYPES = ('Adam', 'SGD')
    FIELDS = (
        'opt_type',
        'lr'
    )

    def __init__(self,
                 *args,
                 opt_type: str = 'Adam',
                 lr: float = 1e-3,
                 **kwargs):
        self.opt_type = opt_type
        self.lr = lr

        super().__init__(*args, **kwargs)


def create_opt(wf=None, opt_config: OptConfig = None):
    if opt_config.opt_type == 'Adam':
        opt_class = pt.optim.Adam
    elif opt_config.opt_type == 'SGD':
        opt_class = pt.optim.SGD
    else:
        raise ValueError(f'Wrong requested optimizer type: {opt_config.opt_type}')

    return opt_class(wf.parameters(), lr=opt_config.lr)
