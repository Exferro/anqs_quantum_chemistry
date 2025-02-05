import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from ....base.constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE
from .multi_head_linear import MultiHeadLinear


class MLP(nn.Module):
    def __init__(self,
                 in_num: int = None,
                 out_num: int = None,
                 val_per_out: int = 1,
                 depth: int = 0,
                 widthes: Union[int, Tuple[int]] = None,
                 activations: Union[None, Callable, Tuple[Callable]] = None,
                 is_made: bool = False,
                 in_group_num: int = None,
                 out_idx_to_in_group_end_idx: Tuple[int] = None,
                 head_num: int = 1,
                 add_bias: bool = True,
                 use_residuals: bool = True,
                 dtype=BASE_REAL_TYPE,
                 is_out_complex: bool = False):
        super(MLP, self).__init__()
        assert dtype in (BASE_REAL_TYPE, BASE_COMPLEX_TYPE)
        self.dtype = dtype
        self.is_out_complex = is_out_complex
        self.is_made = is_made
        self.head_num = head_num
        self.add_bias = add_bias
        self.use_residuals = use_residuals

        self.in_num = in_num
        self.out_num = out_num
        self.val_per_out = val_per_out
        if self.dtype == BASE_REAL_TYPE and self.is_out_complex:
            self.val_per_out = 2 * self.val_per_out

        if in_group_num is not None:
            assert self.is_made
            assert out_idx_to_in_group_end_idx is not None
            assert len(out_idx_to_in_group_end_idx) == in_group_num
            self.in_group_num = in_group_num
            self.out_idx_to_in_group_end_idx = out_idx_to_in_group_end_idx
        else:
            self.in_group_num = in_num
            self.out_idx_to_in_group_end_idx = tuple(in_idx + 1 for in_idx in range(self.in_group_num))

        self.depth = depth

        if isinstance(widthes, tuple):
            assert len(widthes) == self.depth
            self.widthes = widthes
        elif isinstance(widthes, int):
            self.widthes = tuple([widthes] * self.depth)
        else:
            raise RuntimeError(f'Wrong width specifications passed to the MLP: widthes = {widthes}. '
                               f'Should be either int, or tuple of ints.')
        if self.use_residuals:
            assert len(set(self.widthes)) == 1

        self.in_nums = (self.in_num, ) + self.widthes
        self.out_nums = self.widthes
        if self.is_made:
            self.out_nums = self.out_nums + (self.out_num * self.val_per_out, )
        else:
            self.out_nums = self.out_nums + (self.out_num, )

        if self.head_num > 1:
            self.layers = nn.ModuleList([MultiHeadLinear(input_size=self.in_nums[layer_idx],
                                                         output_size=self.out_nums[layer_idx],
                                                         head_num=self.head_num,
                                                         add_bias=self.add_bias,
                                                         dtype=self.dtype)
                                         for layer_idx in range(self.depth + 1)])
        else:
            self.layers = nn.ModuleList([nn.Linear(self.in_nums[layer_idx],
                                                   self.out_nums[layer_idx],
                                                   bias=self.add_bias,
                                                   dtype=self.dtype)
                                         for layer_idx in range(self.depth + 1)])

        # Figuring out activations
        if isinstance(activations, Tuple):
            assert len(activations) == (self.depth + 1)
            self.activations = activations
        elif isinstance(activations, Callable):
            self.activations = (activations, ) * (self.depth + 1)
        elif activations is None:
            if self.dtype == BASE_REAL_TYPE:
                self.activations = (pt.tanh, ) * (self.depth + 1)
            elif self.dtype == BASE_COMPLEX_TYPE:
                self.activations = (lambda x: pt.complex(pt.tanh(x.real,),
                                                         pt.tanh(x.imag)),) * (self.depth + 1)

        # Calculating causal masks required for MADE MLPs
        if self.is_made:
            # self.cond_out_num = out_num
            # self.mid_neuron_allowed_ins = []
            # for layer_idx in range(self.depth):
            #     self.mid_neuron_allowed_ins.append([])
            #     for in_group_idx in range(self.cond_out_num):
            #         self.mid_neuron_allowed_ins[-1] += [self.out_idx_to_in_group_end_idx[in_group_idx] - 1] * (
            #                     self.widthes[layer_idx] // self.cond_out_num + 1 * (
            #                         (self.cond_out_num - in_group_idx - 1) < (self.widthes[layer_idx] % self.cond_out_num)))
            # self.mid_neuron_allowed_ins = pt.tensor(self.mid_neuron_allowed_ins)
            # self.start_mask = pt.ge(pt.unsqueeze(self.mid_neuron_allowed_ins[0], dim=-1),
            #                              pt.unsqueeze(pt.arange(self.in_num), dim=0)).type(self.dtype)
            # self.mid_masks = []
            # for layer_idx in range(1, self.depth):
            #     self.mid_masks.append(pt.ge(pt.unsqueeze(self.mid_neuron_allowed_ins[layer_idx], dim=-1),
            #                                 pt.unsqueeze(self.mid_neuron_allowed_ins[layer_idx - 1], dim=0)).type(self.dtype))
            # self.end_connect = []
            # for in_group_idx in range(self.in_group_num):
            #     self.end_connect += [self.out_idx_to_in_group_end_idx[in_group_idx] - 1]
            # self.end_connect = pt.tensor(self.end_connect)
            # self.end_connect = pt.tile(self.end_connect, (self.val_per_out, 1))
            # self.end_connect = pt.reshape(self.end_connect.T, (-1,))
            #
            # self.end_mask = pt.ge(pt.unsqueeze(self.end_connect, dim=-1),
            #                            pt.unsqueeze(self.mid_neuron_allowed_ins[-1], dim=0)).type(self.dtype)
            #
            # self.masks = (self.start_mask,) + tuple(self.mid_masks) + (self.end_mask,)
            self.mid_neuron_allowed_outs = []
            for layer_idx in range(self.depth):
                self.mid_neuron_allowed_outs.append([])
                for in_group_idx in range(self.out_num):
                    self.mid_neuron_allowed_outs[-1] += [in_group_idx] * (
                            self.widthes[layer_idx] // self.out_num + 1 * (
                            (self.out_num - in_group_idx - 1) < (self.widthes[layer_idx] % self.out_num)))
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

            self.masks = (self.start_mask, ) + tuple(self.mid_masks) + (self.end_mask,)

    def align_input(self, x: pt.Tensor, x_is_base_vec: bool = True):
        if x.shape[-1] == 0:
            x = 0.5 * pt.ones((*x.shape[:-1], self.in_num),
                              dtype=self.dtype,
                              device=x.device)
        if x_is_base_vec:
            aligned_x = 1 - 2 * x.type(self.dtype)
        else:
            aligned_x = x

        if self.head_num > 1:
            aligned_x = pt.tile(pt.unsqueeze(aligned_x, dim=0),
                                (self.head_num, *[1 for _ in aligned_x.shape]))

        return aligned_x

    def forward(self, x):
        assert x.dtype == self.dtype
        if self.head_num > 1:
            assert x.shape[0] == self.head_num
        if self.is_made:
            if x.shape[-1] < self.in_num:
                x = pt.cat((x,
                            pt.zeros((*x.shape[:-1], self.in_num - x.shape[-1]),
                                     dtype=x.dtype,
                                     device=x.device)),
                           dim=-1)
        for layer_idx in range(self.depth + 1):
            if self.use_residuals:
                res_inp = x

            if self.is_made:
                # Apply the mask
                mask = self.masks[layer_idx].to(x.device)
                self.layers[layer_idx].weight.data = self.layers[layer_idx].weight.data * mask

            x = self.layers[layer_idx](x)

            if self.use_residuals:
                if (0 < layer_idx) and (layer_idx < self.depth):
                    x = (x + res_inp)

            x = self.activations[layer_idx](x)

        if self.is_made:
            return pt.reshape(x, (*x.shape[:-1], self.out_num, self.val_per_out))
        else:
            return x
