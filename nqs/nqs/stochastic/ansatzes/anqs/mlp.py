from __future__ import annotations

import torch as pt
from torch import nn

from typing import Callable

from ....base.constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE
from ....infrastructure.nested_data import Config, PatternConfig
from ....base.qubit_grouping import QubitGrouping


class WidthConfig(PatternConfig):
    UNIFORM_PATTERN_FIELD = 'width'
    FIELDS = PatternConfig.FIELDS + (UNIFORM_PATTERN_FIELD,)

    def __init__(self,
                 *args,
                 width: int = 64,
                 **kwargs):
        self.width = width

        super().__init__(*args, **kwargs)


class BiasConfig(PatternConfig):
    UNIFORM_PATTERN_FIELD = 'use_bias'
    FIELDS = PatternConfig.FIELDS + (UNIFORM_PATTERN_FIELD,)

    def __init__(self,
                 *args,
                 use_bias: bool = True,
                 **kwargs):
        self.use_bias = use_bias

        super().__init__(*args, **kwargs)


class ComplexLeakyRelu(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, x):

        return pt.complex(pt.nn.functional.leaky_relu(x.real, negative_slope=0.01),
                          pt.nn.functional.leaky_relu(x.imag, negative_slope=0.01))


class ActivationConfig(PatternConfig):
    ALLOWED_PATTERN_TYPES = PatternConfig.ALLOWED_PATTERN_TYPES + ('sanqs_paper',)
    UNIFORM_PATTERN_FIELD = 'activation'
    FIELDS = PatternConfig.FIELDS + (UNIFORM_PATTERN_FIELD,)
    NON_JSONABLE_FIELDS = (
        'activation',
    )

    def __init__(self,
                 *args,
                 activation: Callable = pt.nn.Tanh,
                 **kwargs):
        self.activation = activation

        super().__init__(*args, **kwargs)

    def create_non_uniform_pattern(self,
                                   depth: int = None):
        if self.pattern_type == 'sanqs_paper':
            return (pt.nn.Tanh, ) + (ComplexLeakyRelu, ) * (depth - 1)
        else:
            raise NotImplementedError


class MLPConfig(Config):
    FIELDS = (
        'depth',
        'width_config',
        'use_res',
        'bias_config',
        'activation_config',
        'activate_last_layer',
    )

    def __init__(self,
                 *args,
                 depth: int = 2,
                 width_config: WidthConfig = None,
                 use_res: bool = True,
                 bias_config: BiasConfig = None,
                 activation_config: ActivationConfig = None,
                 activate_last_layer: bool = False,
                 **kwargs):
        self.depth = depth
        self.width_config = width_config if width_config is not None else WidthConfig()
        self.use_res = use_res
        self.bias_config = bias_config if bias_config is not None else BiasConfig()
        self.activation_config = activation_config if activation_config is not None else ActivationConfig()
        self.activate_last_layer = activate_last_layer

        super().__init__(*args, **kwargs)


class MLP(nn.Module):
    def __init__(self,
                 in_num: int = None,
                 is_made: bool = False,
                 qubit_grouping: QubitGrouping = None,
                 out_num: int = None,
                 dtype=BASE_REAL_TYPE,
                 is_out_complex: bool = False,
                 config: MLPConfig = None):

        super(MLP, self).__init__()
        self.in_num = in_num

        self.is_made = is_made
        self.qubit_grouping = qubit_grouping

        if self.is_made:
            if self.qubit_grouping is None:
                self.out_num = self.in_num
                self.val_per_out = out_num
            else:
                self.out_num = self.qubit_grouping.qudit_num
                self.val_per_out = max(self.qubit_grouping.qudit_dims)
        else:
            self.out_num = out_num
            self.val_per_out = 1

        assert dtype in (BASE_REAL_TYPE, BASE_COMPLEX_TYPE)
        self.dtype = dtype
        self.is_out_complex = is_out_complex
        if self.dtype == BASE_REAL_TYPE and self.is_out_complex:
            self.val_per_out = 2 * self.val_per_out

        self.config = config if config is not None else MLPConfig()
        self.depth = self.config.depth
        self.is_made = is_made

        if self.config.use_res:
            assert self.config.width_config.pattern_type == 'uniform'

        self.width_pattern = self.config.width_config.create_pattern(depth=self.depth)
        self.bias_pattern = self.config.bias_config.create_pattern(depth=self.depth + 1)
        if self.config.activate_last_layer:
            self.activations = pt.nn.ModuleList(tuple(obj() for obj in self.config.activation_config.create_pattern(depth=self.depth + 1)))
        else:
            self.activations = pt.nn.ModuleList(tuple(obj() for obj in self.config.activation_config.create_pattern(depth=self.depth))
                                                + (pt.nn.Identity(),))

        if self.qubit_grouping is None:
            self.in_group_num = self.in_num
            self.out_idx_to_in_group_end_idx = tuple(in_idx + 1 for in_idx in range(self.in_group_num))
        else:
            self.in_group_num = self.qubit_grouping.qudit_num
            self.out_idx_to_in_group_end_idx = self.qubit_grouping.qudit_ends

        self.in_nums = (self.in_num, ) + self.width_pattern
        self.out_nums = self.width_pattern
        if self.is_made:
            self.out_nums = self.out_nums + (self.out_num * self.val_per_out, )
        else:
            self.out_nums = self.out_nums + (self.out_num, )

        self.layers = nn.ModuleList([nn.Linear(self.in_nums[layer_idx],
                                               self.out_nums[layer_idx],
                                               bias=self.bias_pattern[layer_idx],
                                               dtype=self.dtype)
                                     for layer_idx in range(self.depth + 1)])

        # Calculating causal masks required for MADE MLPs
        self.mid_neuron_allowed_outs = []
        for layer_idx in range(self.depth):
            self.mid_neuron_allowed_outs.append([])
            for in_group_idx in range(self.out_num):
                self.mid_neuron_allowed_outs[-1] += [in_group_idx] * (
                        self.width_pattern[layer_idx] // self.out_num + 1 * (
                        (self.out_num - in_group_idx - 1) < (self.width_pattern[layer_idx] % self.out_num)))
        self.mid_neuron_allowed_outs = pt.tensor(self.mid_neuron_allowed_outs)

        self.start_connect = []
        for in_group_idx in range(self.in_group_num):
            if in_group_idx == 0:
                self.start_connect = self.start_connect + [in_group_idx] * (self.out_idx_to_in_group_end_idx[in_group_idx])
            else:
                self.start_connect = self.start_connect + [in_group_idx] * (
                            self.out_idx_to_in_group_end_idx[in_group_idx] - self.out_idx_to_in_group_end_idx[
                        in_group_idx - 1])

        self.start_connect = pt.tensor(self.start_connect)

        self.start_mask = pt.greater(pt.unsqueeze(self.mid_neuron_allowed_outs[0], dim=-1),
                                pt.unsqueeze(self.start_connect, dim=0)).type(self.dtype)
        self.mid_masks = []
        for layer_idx in range(1, self.depth):
            self.mid_masks.append(pt.ge(pt.unsqueeze(self.mid_neuron_allowed_outs[layer_idx], dim=-1),
                                        pt.unsqueeze(self.mid_neuron_allowed_outs[layer_idx - 1], dim=0)).type(self.dtype))

        self.end_connect = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(self.in_group_num), dim=-1), (1, self.val_per_out)),
                                 (-1,))
        self.end_mask = pt.ge(pt.unsqueeze(self.end_connect, dim=-1),
                         pt.unsqueeze(self.mid_neuron_allowed_outs[-1], dim=0)).type(self.dtype)

        self.made_masks = (self.start_mask,) + tuple(self.mid_masks) + (self.end_mask,)

    def align_input(self, x: pt.Tensor, x_is_base_vec: bool = True):
        if x.shape[-1] == 0:
            x = 0.5 * pt.ones((*x.shape[:-1], self.in_num),
                              dtype=self.dtype,
                              device=x.device)
        if x_is_base_vec:
            aligned_x = 1 - 2 * x.type(self.dtype)
        else:
            aligned_x = x

        return aligned_x

    def forward(self, x):
        assert x.dtype == self.dtype
        if self.is_made:
            if x.shape[-1] < self.in_num:
                x = pt.cat((x,
                            pt.zeros((*x.shape[:-1], self.in_num - x.shape[-1]),
                                     dtype=x.dtype,
                                     device=x.device)),
                           dim=-1)
        for layer_idx in range(self.depth + 1):
            if self.config.use_res:
                res_inp = x

            if self.is_made:
                # Apply the mask
                mask = self.made_masks[layer_idx].to(x.device)
                self.layers[layer_idx].weight.data = self.layers[layer_idx].weight.data * mask

            x = self.layers[layer_idx](x)

            if self.config.use_res:
                if (0 < layer_idx) and (layer_idx < self.depth):
                    x = (x + res_inp)

            x = self.activations[layer_idx](x)

        if self.is_made:
            return pt.reshape(x, (*x.shape[:-1], self.out_num, self.val_per_out))
        else:
            return x
