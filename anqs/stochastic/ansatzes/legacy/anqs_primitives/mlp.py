import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE


class MLP(nn.Module):
    """
    A class for a multi-layer perceptron specified by the number of inputs,
    number of outputs, number of hidden layers (aka depth) and numbed of neurons
    inside every hidden layer (aka width).

    In this package we will use such MLPs as conditional subnetworks of an NNQS
    wave function.
    """
    def __init__(self,
                 in_num: int = None,
                 out_num: int = None,
                 depth: int = 0,
                 width: int = None,
                 activations: Union[Tuple[Callable], None] = None,
                 dtype=BASE_REAL_TYPE):
        super(MLP, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.depth = depth
        self.width = width
        self.dtype = dtype

        # Creating layers
        self.in_nums = [self.virt_in_num] + [width] * self.depth
        self.out_nums = [width] * self.depth + [out_num]
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.in_nums[layer_idx],
                                                             self.out_nums[layer_idx],
                                                             bias=False,
                                                             dtype=self.dtype))
                                     for layer_idx in range(self.depth + 1)])
        self.activations = None
        if activations is not None:
            assert len(activations) == self.depth + 1
            self.activations = activations

    @property
    def virt_in_num(self):
        """
        A property required to account for the first unconditional MLP in the NAQS:
        it takes no inputs, but we model it as having a (constant) input of size 1.
        Hence, if we claim to create an MLP with 0 inputs, in fact we silently create
        an MLP with 1 input.
        """
        return self.in_num if self.in_num > 0 else self.in_num + 1

    def forward(self, x):
        assert x.shape[-1] == self.in_num
        if x.shape[-1] == 0:
            x = 1 * pt.ones((*x.shape[:-1], self.virt_in_num),
                            dtype=x.dtype,
                            device=x.device)
        for layer_idx in range(self.depth + 1):
            x = self.layers[layer_idx](x)
            if self.activations is not None:
                x = self.activations[layer_idx](x)
            else:
                if x.dtype == BASE_REAL_TYPE:
                    if layer_idx < self.depth:
                        x = nn.functional.leaky_relu(x)
                elif x.dtype == BASE_COMPLEX_TYPE:
                    if layer_idx < self.depth:
                        if layer_idx == 0:
                            x = pt.tanh(x)
                        else:
                            x = pt.complex(nn.functional.leaky_relu(x.real, negative_slope=0.01),
                                           nn.functional.leaky_relu(x.imag, negative_slope=0.01))
                else:
                    raise RuntimeError(f'Unsupported dtype: {x.dtype}')

        return x
