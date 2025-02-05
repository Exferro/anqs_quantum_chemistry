import torch as pt

from ..base.constants import INT_32_TYPE, INT_64_TYPE
from ..base.constants import BIT_DEPTH_TO_INT_TYPE

M1 = {
    32: pt.tensor(0x55555555, dtype=INT_32_TYPE),
    64: pt.tensor(0x5555555555555555, dtype=INT_64_TYPE),
}
M2 = {
    32: pt.tensor(0x33333333, dtype=INT_32_TYPE),
    64: pt.tensor(0x3333333333333333, dtype=INT_64_TYPE),
}
M4 = {
    32: pt.tensor(0x0f0f0f0f, dtype=INT_32_TYPE),
    64: pt.tensor(0x0f0f0f0f0f0f0f0f, dtype=INT_64_TYPE),
}
H01 = {
    32: pt.tensor(0x01010101, dtype=INT_32_TYPE),
    64: pt.tensor(0x0101010101010101, dtype=INT_64_TYPE),
}


def popcount(x, *, bit_depth=64):
    x = pt.clone(x)
    x = x.type(BIT_DEPTH_TO_INT_TYPE[bit_depth])
    x = x.sub_((pt.bitwise_right_shift(x, 1)).bitwise_and_(M1[bit_depth]))
    #x = x.sub_(pt.bitwise_and(pt.bitwise_right_shift(x, 1), M1[bit_depth]))
    x = pt.bitwise_and(x, M2[bit_depth]).add_(pt.bitwise_right_shift(x, 2).bitwise_and_(M2[bit_depth]))
    #x = pt.bitwise_and(pt.add(x, pt.bitwise_right_shift(x, 4)), M4[bit_depth])
    x = (x.add_(pt.bitwise_right_shift(x, 4))).bitwise_and_(M4[bit_depth])

    #return pt.bitwise_right_shift(pt.mul(x, H01[bit_depth]), bit_depth - 8).type(BIT_DEPTH_TO_INT_TYPE[bit_depth])
    return ((x.mul_(H01[bit_depth])).bitwise_right_shift_(bit_depth - 8)).type(BIT_DEPTH_TO_INT_TYPE[bit_depth])
