import torch as pt

BASE_INT_TYPE = pt.int64
BASE_REAL_TYPE = pt.double
BASE_COMPLEX_TYPE = pt.cdouble

INT_32_TYPE = pt.int32
INT_64_TYPE = pt.int64

BIT_DEPTH_TO_INT_TYPE = {
    32: INT_32_TYPE,
    64: INT_64_TYPE,
}

NEGINF = pt.tensor(-float('inf'), dtype=BASE_REAL_TYPE)
