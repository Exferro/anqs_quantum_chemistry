import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE


class MADEMLP(nn.Module):
    def __init__(self,
                 in_num: int = None,
                 out_num: int = None,
                 depth: int = 0,
                 width: int = None,
                 activations: Union[Tuple[Callable], None] = None,
                 dtype=BASE_REAL_TYPE):
        super(MADEMLP, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.depth = depth
        self.width = width
        self.dtype = dtype

        self.in_nums = [self.in_num] + [self.width] * self.depth
        self.out_nums = [self.width] * self.depth + [self.out_num * self.in_num]
        self.layers = nn.ModuleList([nn.Linear(self.in_nums[layer_idx],
                                               self.out_nums[layer_idx],
                                               bias=True,
                                               dtype=self.dtype)
                                     for layer_idx in range(self.depth + 1)])

        self.cond_in_num = self.in_num - 1
        self.mid_neuron_allowed_ins = []
        for in_idx in range(self.cond_in_num):
            self.mid_neuron_allowed_ins += [in_idx] * (self.width // self.cond_in_num + 1 * ((self.cond_in_num - in_idx - 1) < (width % self.cond_in_num)))
        self.mid_neuron_allowed_ins = pt.tensor(self.mid_neuron_allowed_ins)
        self.start_mask = pt.ge(pt.unsqueeze(self.mid_neuron_allowed_ins, dim=-1),
                                pt.unsqueeze(pt.arange(self.in_num), dim=0)).type(self.dtype)
        self.mid_mask = pt.ge(pt.unsqueeze(self.mid_neuron_allowed_ins, dim=-1),
                              pt.unsqueeze(self.mid_neuron_allowed_ins, dim=0)).type(self.dtype)
        self.end_mask = pt.greater(pt.unsqueeze(pt.arange(self.in_num * self.out_num) // self.out_num, dim=-1),
                                   pt.unsqueeze(self.mid_neuron_allowed_ins, dim=0)).type(self.dtype)

        self.start_mask = self.start_mask
        self.mid_mask = self.mid_mask
        self.end_mask = self.end_mask

        self.masks = [self.start_mask] + [self.mid_mask] * (self.depth - 1) + [self.end_mask]

        self.activations = None
        if activations is not None:
            assert len(activations) == self.depth + 1
            self.activations = activations

    def forward(self, x):
        if x.shape[-1] < self.in_num:
            x = pt.cat((x,
                        pt.zeros((*x.shape[:-1], self.in_num - x.shape[-1]),
                                 dtype=x.dtype,
                                 device=x.device)),
                       dim=-1)
        for layer_idx in range(self.depth + 1):
            x_inp = x
            # Apply the mask
            self.masks[layer_idx] = self.masks[layer_idx].to(x.device)
            self.layers[layer_idx].weight.data = self.layers[layer_idx].weight.data * self.masks[layer_idx]
            x = self.layers[layer_idx](x)
            if (0 < layer_idx) and (layer_idx < self.depth):
                x = x + x_inp
            if self.activations is not None:
                x = self.activations[layer_idx](x)
            else:
                if x.dtype == BASE_REAL_TYPE:
                    x = pt.tanh(x)
                elif x.dtype == BASE_COMPLEX_TYPE:
                    if layer_idx < self.depth:
                        if layer_idx == 0:
                            x = pt.tanh(x)
                        else:
                            x = pt.complex(nn.functional.leaky_relu(x.real, negative_slope=0.01),
                                           nn.functional.leaky_relu(x.imag, negative_slope=0.01))
                else:
                    raise RuntimeError(f'Unsupported dtype: {x.dtype}')

        return pt.reshape(x, (*x.shape[:-1], self.in_num, self.out_num))

