import torch as pt
from torch import nn

ACTIVATION_TYPES = ('tanh', 'leaky_relu', 'relu')
ACTIVATION_COMBOS = (None, 'default_log_abs_combo', 'default_phase_combo')


def activation_combo2activations(activation_combo: str = None,
                                 depth: int = None):
    assert activation_combo in ACTIVATION_COMBOS
    if activation_combo == 'default_log_abs_combo':
        activations = tuple(lambda x: pt.tanh(x) for _ in range(depth + 1))
    elif activation_combo == 'default_phase_combo':
        activations = tuple(lambda x: pt.tanh(x) for _ in range(depth)) + (lambda x: x,)
    elif activation_combo is None:
        return None
    else:
        raise RuntimeError(f'Wrong activation combo: {activation_combo}')

    return activations


def real_domain_tanh(x):
    re = x[..., :x.shape[-1] // 2]
    im = x[..., x.shape[-1] // 2:]
    new_re = pt.sinh(2 * re) / (pt.cosh(2 * re) + pt.cos(2 * im) + 5 * 1e-3)
    new_im = pt.sin(2 * im) / (pt.cosh(2 * re) + pt.cos(2 * im) + 5 * 1e-3)
    
    return pt.cat([new_re, new_im], dim=-1)

def real_domain_tanh_layer_norm(x):    
    x = real_domain_tanh(x)
    
    return (x - pt.mean(x, dim=-1, keepdim=True)) / pt.sqrt(pt.var(x, dim=-1, keepdim=True) + 1e-5)
