import torch as pt
import torch.nn as nn
import math


class MultiHeadLinear(nn.Module):
    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 head_num: int = None,
                 add_bias: bool = True,
                 dtype=None):
        super(MultiHeadLinear, self).__init__()
        self.dtype = dtype
        self.add_bias = add_bias

        a = math.sqrt(1 / input_size)
        # initialize weights
        self.weight = pt.nn.Parameter(pt.zeros(head_num, output_size, input_size, dtype=self.dtype))
        pt.nn.init.uniform_(self.weight, -a, a)
        if self.add_bias:
            self.bias = pt.nn.Parameter(pt.zeros(head_num, 1, output_size, dtype=self.dtype))
            pt.nn.init.uniform_(self.bias, -a, a)

    def forward(self, x):
        x = pt.bmm(x, pt.transpose(self.weight, -1, -2))
        if self.add_bias:
            x = x + self.bias

        return x
